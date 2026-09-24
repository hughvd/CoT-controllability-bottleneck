"""Bounded, resumable Qwen token-prefix generation on one Modal B200.

modal run --detach controllability/modal_token_prefill.py --phase dispatch
modal run controllability/modal_token_prefill.py --phase collect
"""
from pathlib import Path
import json
import time
import modal

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "results/token_prefill/qwen32_v1"
image = (modal.Image.from_registry("nvidia/cuda:13.0.2-devel-ubuntu24.04", add_python="3.12")
    .entrypoint([]).pip_install("vllm==0.23.0", "transformers==5.12.1", "huggingface_hub")
    .env({"HF_HOME": "/cache", "HF_HUB_DISABLE_XET": "1", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_USE_FLASHINFER_SAMPLER": "0"})
    .add_local_file(HERE / "token_prefill.py", "/root/token_prefill.py"))
app = modal.App("qwen-token-prefill")
cache = modal.Volume.from_name("ban-sweep-hf-cache")
volume = modal.Volume.from_name("cot-token-prefill-out", create_if_missing=True)
TERMINAL = {"complete", "failed", "budget_stop", "projection_exceeds_budget"}


@app.function(image=image, gpu="B200", cpu=8, memory=131072, timeout=7200,
              volumes={"/cache": cache.read_only(), "/out": volume},
              retries=0, max_containers=1, scaledown_window=2)
def generate(manifest, budget_seconds, attempt_id):
    started = time.monotonic()
    import sys
    import shutil
    from concurrent.futures import ThreadPoolExecutor
    sys.path.insert(0, "/root")
    import token_prefill as tp
    import vllm
    import torch
    import transformers
    from vllm import LLM, SamplingParams
    tp.validate_manifest(manifest)
    assert tp.file_hash(tp.__file__) == manifest["code_sha256"]
    assert manifest["model"]["adapter"] == "qwen3_think"
    root = Path("/out") / manifest["run"] / manifest["manifest_sha256"]
    root.mkdir(parents=True, exist_ok=True)
    tp.immutable_json(root / "manifest.json", manifest)
    report = dict(status="starting", attempt_id=attempt_id, started_at=time.time(), batches=[],
                  manifest_sha256=manifest["manifest_sha256"], runner_sha256=tp.file_hash(__file__),
                  vllm=vllm.__version__, torch=torch.__version__, transformers=transformers.__version__,
                  gpu=torch.cuda.get_device_name(), max_num_seqs=128, batch_prompts=128,
                  enforce_eager=False)
    def commit():
        report["elapsed_seconds"] = time.monotonic() - started
        tp.atomic_json(root / "report.json", report)
        volume.commit()
    commit()
    try:
        known = {r["request_id"]: r for r in manifest["requests"]}
        done = set()
        for path in root.glob("batch_*.json"):
            b = json.loads(path.read_text())
            assert b["manifest_sha256"] == manifest["manifest_sha256"]
            for row in b["outputs"]:
                key = (row["request_id"], row["sample"])
                assert key not in done and row["prompt_token_ids"] == known[row["request_id"]]["prompt_token_ids"]
                assert row["sample"] in range(manifest["sampling"]["n"])
                done.add(key)
        report["reused_outputs"] = len(done)
        pending = []
        for r in manifest["requests"]:
            count = sum((r["request_id"], s) in done for s in range(manifest["sampling"]["n"]))
            assert count in [0, manifest["sampling"]["n"]], "Partial request in committed batch"
            if not count:
                pending.append(r)
        if not pending:
            report["status"] = "complete"
            return report
        model = manifest["model"]
        snapshot = Path("/cache/hub") / ("models--" + model["id"].replace("/", "--")) / "snapshots" / model["revision"]
        for name, expected in manifest["tokenizer_hashes"].items():
            assert tp.file_hash(snapshot / name) == expected, name
        staged = Path("/tmp/token-prefill-model")
        staged.mkdir(exist_ok=True)
        def copy(src):
            with src.open("rb") as a, (staged / src.name).open("wb") as b:
                shutil.copyfileobj(a, b, 16 * 1024 * 1024)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(copy, [p for p in snapshot.iterdir() if p.is_file()]))
        print("Model weights staged", flush=True)
        llm = LLM(model=str(staged), tokenizer=str(staged), dtype="bfloat16",
                  max_model_len=manifest["max_model_len"], max_num_seqs=128,
                  max_num_batched_tokens=8192, gpu_memory_utilization=.92,
                  enable_prefix_caching=True, enforce_eager=False, seed=20260915)
        tok = llm.get_tokenizer()
        assert tok.encode("</think>", add_special_tokens=False) == [manifest["protocol"]["think_close_id"]]
        # Test representative full prompts against local preparation.
        for r in pending[:4]:
            text = tok.apply_chat_template([{"role": "user", "content": r["user_text"]}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True) + "<think>\n"
            assert tok.encode(text, add_special_tokens=False) == r["opening_token_ids"]
        report["startup_seconds"] = time.monotonic() - started
        smoke = [r for r in pending if r["request_id"] in manifest["smoke_ids"]]
        rest = [r for r in pending if r["request_id"] not in manifest["smoke_ids"]]
        # Keep the original stress subset and add evenly spaced pending requests
        # before observing outcomes. Benchmark the actual production batch size;
        # the 24-prompt stress-only pilot overweights two difficult questions.
        extra_count = min(max(0, 128 - len(smoke)), len(rest))
        extra_ids = {rest[i * len(rest) // extra_count]["request_id"] for i in range(extra_count)}
        smoke += [r for r in rest if r["request_id"] in extra_ids]
        rest = [r for r in rest if r["request_id"] not in extra_ids]
        report["pilot_request_ids"] = [r["request_id"] for r in smoke]
        batches = [smoke] + [rest[i:i + 128] for i in range(0, len(rest), 128)]
        cells = {c["cell_id"]: c for c in manifest["cells"]}
        per_request_seconds = None
        for i, rows in enumerate(batches):
            remaining = budget_seconds - (time.monotonic() - started)
            if remaining < max(180, (per_request_seconds or 0) * len(rows) * 1.5 + 60):
                report["status"] = "budget_stop"
                break
            params = [SamplingParams(**manifest["sampling"], seed=r["sampling_seed"],
                                      skip_special_tokens=False) for r in rows]
            t = time.monotonic()
            outputs = llm.generate([{"prompt_token_ids": r["prompt_token_ids"]} for r in rows], params, use_tqdm=False)
            assert len(outputs) == len(rows)
            records, empty, truncated = [], 0, 0
            for r, output in zip(rows, outputs):
                assert list(output.prompt_token_ids) == r["prompt_token_ids"]
                assert len(output.outputs) == manifest["sampling"]["n"]
                for sample, answer in enumerate(output.outputs):
                    ids = list(answer.token_ids)
                    check = tp.measure(tok, r, ids, manifest["protocol"], cells[r["cell_id"]]["behavior"])
                    empty += check["empty_reasoning"]
                    truncated += answer.finish_reason == "length"
                    records.append(dict(request_id=r["request_id"], sample=sample,
                        prompt_token_ids=r["prompt_token_ids"], output_token_ids=ids,
                        raw_text=answer.text, finish_reason=answer.finish_reason, stop_reason=answer.stop_reason))
            elapsed = time.monotonic() - t
            batch_id = tp.digest([r["request_id"] for r in rows])
            tp.immutable_json(root / f"batch_{batch_id}.json",
                dict(manifest_sha256=manifest["manifest_sha256"], outputs=records))
            report["batches"].append(dict(batch=batch_id, outputs=len(records), seconds=elapsed,
                                           empty=empty, truncated=truncated))
            report["status"] = "generating"
            commit()
            print(report["batches"][-1], flush=True)
            per_request_seconds = max(per_request_seconds or 0, elapsed / len(rows))
            if i == 0:
                if empty / len(records) > .1 or truncated / len(records) > .1:
                    raise ValueError("Pilot exceeds 10% empty reasoning or truncation; inspect saved outputs")
                projection = elapsed / len(rows) * len(rest) * 1.5 + 180
                report["projected_remaining_seconds"] = projection
                if projection > budget_seconds - (time.monotonic() - started):
                    report["status"] = "projection_exceeds_budget"
                    break
        else:
            report["status"] = "complete"
    except Exception as e:
        report["status"] = "failed"
        report["error_type"] = type(e).__name__
        raise
    finally:
        commit()
    return report


@app.local_entrypoint()
def main(phase: str = "status", root: str = str(DEFAULT_ROOT)):
    import sys
    sys.path.insert(0, str(HERE))
    import token_prefill as tp
    root = Path(root)
    m = json.loads((root / "manifest.json").read_text())
    tp.validate_manifest(m)
    remote = f'{m["run"]}/{m["manifest_sha256"]}'
    ledger_path = root / "gpu_ledger.json"
    if phase in {"status", "collect"}:
        report = json.loads(b"".join(volume.read_file(remote + "/report.json")))
        print({k: report.get(k) for k in ["status", "elapsed_seconds", "reused_outputs", "batches"]})
        if phase == "collect":
            for item in volume.listdir(remote):
                if item.path.endswith(".json"):
                    destination = root / "remote" / Path(item.path).name
                    destination.parent.mkdir(exist_ok=True)
                    if destination.exists() and destination.name.startswith("batch_"):
                        continue
                    tmp = destination.with_suffix(".tmp")
                    with tmp.open("wb") as f:
                        for chunk in volume.read_file(item.path):
                            f.write(chunk)
                    tmp.replace(destination)
            if report["status"] in TERMINAL:
                ledger = json.loads(ledger_path.read_text())
                entry = next(a for a in ledger["attempts"] if a["attempt_id"] == report["attempt_id"])
                entry.update(status=report["status"], accounted_seconds=report["elapsed_seconds"] + 60)
                tp.atomic_json(ledger_path, ledger)
        return
    if phase != "dispatch":
        raise ValueError("phase must be dispatch, status, or collect")
    assert tp.file_hash(tp.__file__) == m["code_sha256"]
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else dict(
        max_seconds=7200, conservative_usd_per_hour=8., max_estimated_usd=12., attempts=[])
    if any(a["status"] in {"reserved", "spawned"} for a in ledger["attempts"]):
        raise ValueError("A worker reservation is active; collect/reconcile it before dispatching")
    if any(a["status"] == "complete" for a in ledger["attempts"]):
        print("Already complete; no worker launched")
        return
    remaining = ledger["max_seconds"] - sum(a["accounted_seconds"] for a in ledger["attempts"])
    ledger.setdefault("max_estimated_usd", 12.)
    spent = sum(a["accounted_seconds"] / 3600 * a.get("usd_per_hour", ledger["conservative_usd_per_hour"])
                for a in ledger["attempts"])
    remaining = min(remaining, (ledger["max_estimated_usd"] - spent) / 8. * 3600)
    if remaining < 300:
        raise ValueError("Cumulative GPU runtime cap exhausted")
    attempt = dict(attempt_id=tp.digest([time.time(), m["manifest_sha256"]]),
                   started_at=time.time(), reserved_seconds=remaining, status="reserved",
                   gpu="B200", usd_per_hour=8.)
    ledger["attempts"].append(attempt)
    tp.atomic_json(ledger_path, ledger)
    call = generate.with_options(timeout=int(remaining)).spawn(m, remaining - 60, attempt["attempt_id"])
    attempt.update(status="spawned", call_id=call.object_id)
    tp.atomic_json(ledger_path, ledger)
    print("Dispatched:", call.object_id)
