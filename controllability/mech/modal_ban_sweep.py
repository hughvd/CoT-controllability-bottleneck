"""Modal harness for the ban-sampling strategy sweep (GPU trace-gen only; grade local).

Runs mech/ban_sweep.py on ONE warm H100-80GB (tp=1 fits a 32B in bf16 with ~16G KV
headroom), looping over all requested variants in a single container so the 64G model
loads ONCE. Weights persist in a HF-cache Volume (first run downloads ~64G via
hf_transfer; later runs reuse). Full traces persist to an output Volume; a slim copy
(no token_ids) is returned so the local entrypoint writes the local traces.jsonl that
mech/olmo_interventions_grade.py grades.

  # first batch: depth ladder, no gate, OLMo-3.1-32B-Think (entrenched final = hardest)
  modal run controllability/mech/modal_ban_sweep.py \
      --variants "baseline ban1 ban2 ban3 ban4 ban5" --runtag sweep_olmo31 --subset 15

  # adaptive follow-up: add gate variants near the winner (baseline+ban1..5 cached -> skipped)
  modal run controllability/mech/modal_ban_sweep.py \
      --variants "ban3_g95 ban3_g99 ban4_g95 ban4_g99" --runtag sweep_olmo31 --subset 15

OLMo-3 needs the bleeding-edge stack (vllm 0.23 / transformers 5.12, olmo3 supported by
both). Offline LLM() -> no server, so the fastapi/serving pins don't apply. xet disabled
to avoid the hf-xet D-state load hang seen on the pods.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.23.0",
        "transformers==5.12.1",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache",
          # FlashInfer's top-p/top-k sampler JIT-compiles a CUDA kernel (needs nvcc, absent
          # in the slim image). Route sampling to the native Torch sampler instead.
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_file(HERE / "common.py", "/root/mech/common.py")
    .add_local_file(HERE / "ban_sweep.py", "/root/mech/ban_sweep.py")
    .add_local_file(HERE / "data" / "olmo_rl" / "prompts.jsonl",
                    "/root/mech/data/olmo_rl/prompts.jsonl")
    # full-26 ctrl prompt sets (built locally by build_ctrl26_prompts.py before launch)
    .add_local_dir(HERE / "data" / "ctrl26", "/root/mech/data/ctrl26")
)

app = modal.App("ban-sweep")
hf_cache = modal.Volume.from_name("ban-sweep-hf-cache", create_if_missing=True)
out_vol = modal.Volume.from_name("ban-sweep-out", create_if_missing=True)


@app.function(
    image=image,
    gpu="H100",
    volumes={"/cache": hf_cache, "/out": out_vol},
    timeout=8 * 3600,   # OLMo ban4 traces run long (entrenched model rambles to max_tokens)
)
def run_sweep(model: str, runtag: str, variants: list[str], subset: int,
              n_samples: int, conditions: list[str], prompts: str) -> list[dict]:
    import sys
    sys.path.insert(0, "/root/mech")
    import ban_sweep as bs

    out_root = "/out"
    sys.argv = [
        "ban_sweep.py", "--model", model, "--runtag", runtag,
        "--variants", *variants, "--conditions", *conditions,
        "--subset", str(subset), "--n-samples", str(n_samples),
        "--tp", "1", "--prompts", prompts,
        "--out-root", out_root,
    ]
    bs.main()
    out_vol.commit()

    traces = bs.load_jsonl(Path(out_root) / runtag / "traces.jsonl")
    slim = [{k: v for k, v in t.items() if k != "output_token_ids"} for t in traces]
    print(f"[modal] returning {len(slim)} slim traces for local grading")
    return slim


@app.local_entrypoint()
def main(variants: str, runtag: str = "sweep_olmo31",
         model: str = "allenai/Olmo-3.1-32B-Think",
         subset: int = 15, n_samples: int = 2, conditions: str = "instruction_only",
         prompts: str = "/root/mech/data/olmo_rl/prompts.jsonl",
         out_local: str = "olmo_rl_interventions"):
    vlist = variants.split()
    clist = conditions.split()
    print(f"[local] launching sweep runtag={runtag} model={model}")
    print(f"[local] variants={vlist} subset={subset} n_samples={n_samples} "
          f"conditions={clist} prompts={prompts}")
    slim = run_sweep.remote(model, runtag, vlist, subset, n_samples, clist, prompts)

    out = HERE / "data" / out_local / runtag / "traces.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for t in slim:
            f.write(json.dumps(t) + "\n")
    print(f"[local] wrote {len(slim)} traces -> {out}")
    print(f"[local] grade with:  python mech/olmo_interventions_grade.py --tag {runtag}")
