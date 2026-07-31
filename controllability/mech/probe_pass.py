"""Overnight job 1 — fate-probe activations (base Qwen3-32B, HF teacher-forced).

For EVERY existing mech trace, one truncated forward through the base model saving
residual-stream states at the positions a compliance probe will read:
  * end-of-prompt (the state that emits generation token 0 — the "<think>-token"
    state; identical across the 10 samples of a prompt, saved once per prompt)
  * after generated token g for g in GEN_POS (per trace — these diverge across
    samples once sampling starts) plus the end of the first CoT sentence

The forward is cut at max(GEN_POS)+1 / first-sentence-end tokens of generation, so
this is ~10x cheaper than the full entropy pass. Probe training happens locally
(analysis/probe_analysis.py): logistic regression on each (layer, position), label =
trace compliance, question-grouped CV -> AUC vs position ("when is fate readable?").

  python -u mech/probe_pass.py [--limit N]
  # organism side (4a probe-transfer): same positions, organism model + its traces
  python -u mech/probe_pass.py --model artifacts/organism_2_v34_ctrl_all/merged \
      --traces data/kl/org_traces.jsonl --out-root data/probe_org

Outputs {out-root}/prompt/{pid}.npz (eop state) + {out-root}/trace/{tid}.npz. Resumable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from common import DATA, MODEL_HF, PROMPTS_PATH, TRACES_PATH, load_jsonl

sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))
from common_analysis import content_start, first_sentence_end, pieces, tid_fname

# hidden_states index l is the output of block l-1 (0 = embeddings); Qwen3-32B has 64
# blocks. We save the outputs of these blocks (spread + last).
LAYERS = [8, 16, 24, 32, 40, 48, 56, 63]
GEN_POS = [0, 1, 2, 4, 8, 16, 32, 64, 128]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=MODEL_HF, help="HF id or merged model dir")
    ap.add_argument("--traces", default=str(TRACES_PATH))
    ap.add_argument("--out-root", default=str(DATA / "probe"))
    args = ap.parse_args()
    out_root = Path(args.out_root)
    probe_prompt_dir = out_root / "prompt"
    probe_trace_dir = out_root / "trace"

    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    model.eval()

    prompts = {p["prompt_id"]: p for p in load_jsonl(PROMPTS_PATH)}
    traces = load_jsonl(Path(args.traces))
    if args.limit:
        traces = traces[:args.limit]
    probe_prompt_dir.mkdir(parents=True, exist_ok=True)
    probe_trace_dir.mkdir(parents=True, exist_ok=True)
    lsel = [l + 1 for l in LAYERS]   # -> hidden_states tuple indices

    n_new = 0
    for k, t in enumerate(traces):
        out_path = probe_trace_dir / f"{tid_fname(t['trace_id'])}.npz"
        if out_path.exists() or not t["output_token_ids"]:
            continue
        p = prompts[t["prompt_id"]]
        pcs = pieces(t["output_token_ids"])
        cs = content_start(pcs)
        s1 = first_sentence_end(pcs, cs)
        gen_len = len(t["output_token_ids"])
        t_cut = min(gen_len, max(GEN_POS[-1] + 1, s1 + 2))
        P = len(p["prompt_token_ids"])

        ids = torch.tensor([p["prompt_token_ids"] + t["output_token_ids"][:t_cut]],
                           device=model.device)
        with torch.no_grad():
            hs = model(ids, output_hidden_states=True).hidden_states
        # [n_layers, T, d] for the selected layers
        H = torch.stack([hs[i][0] for i in lsel]).to(torch.float16)

        pp = probe_prompt_dir / f"{tid_fname(t['prompt_id'])}.npz"
        if not pp.exists():
            np.savez_compressed(pp, eop=H[:, P - 1].cpu().numpy(),
                                layers=np.array(LAYERS), prompt_len=P)

        keep = sorted({g for g in GEN_POS if g < t_cut} | {min(cs, t_cut - 1),
                                                           min(s1, t_cut - 1)})
        np.savez_compressed(
            out_path,
            hs=H[:, [P + g for g in keep]].cpu().numpy(),   # [n_layers, n_pos, d]
            gen_positions=np.array(keep), layers=np.array(LAYERS),
            content_start=cs, sent1_end=s1, gen_len=gen_len, prompt_len=P)
        del hs, H
        n_new += 1
        if (k + 1) % 200 == 0:
            print(f"[probe] {k + 1}/{len(traces)} traces ({n_new} new)")
    print(f"[probe] done: {len(traces)} traces, {n_new} new -> {probe_trace_dir}")


if __name__ == "__main__":
    main()
