"""Sequential, budget-limited H100 rescoring. Never generates or grades responses.

modal run controllability/mech/modal_suite_rank_probe.py --run opening_suite_v1
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.23.0", "transformers==5.12.1", "numpy", "huggingface_hub", "hf_transfer")
         .pip_install("accelerate==1.12.0")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HUB_DISABLE_XET": "1", "HF_HOME": "/cache",
               "VLLM_USE_FLASHINFER_SAMPLER": "0", "TOKENIZERS_PARALLELISM": "false"})
         .add_local_file(HERE / "rank_probe.py", "/root/rank_probe.py")
         .add_local_file(HERE / "suite_rank_probe.py", "/root/suite_rank_probe.py"))
app = modal.App("cot-suite-ranks")
cache = modal.Volume.from_name("ban-sweep-hf-cache")
results_volume = modal.Volume.from_name("cot-suite-ranks-out", create_if_missing=True)


@app.function(image=image, gpu="H100", cpu=4, memory=131072,
              volumes={"/cache": cache.read_only(), "/out": results_volume},
              timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def rescore(manifest: dict, tag: str, max_seconds: float):
    started = time.monotonic()
    import sys
    sys.path.insert(0, "/root")
    import suite_rank_probe as rp
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if rp.file_hash(Path(rp.base.__file__)) != manifest["base_scorer_sha256"]:
        raise ValueError("Base scorer changed")
    if rp.file_hash(Path(rp.__file__)) != manifest["scorer_sha256"]:
        raise ValueError("Uploaded scorer differs from immutable manifest")
    if rp.digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}) != manifest["manifest_sha256"]:
        raise ValueError("Manifest content changed")
    out = Path("/out") / manifest["run"] / manifest["manifest_sha256"] / tag
    out.mkdir(parents=True, exist_ok=True)
    rp.immutable_json(out / "manifest.json", manifest)
    fp = out / "pairs.jsonl"
    done = rp.load_results(fp, manifest["manifest_sha256"])
    if fp.exists():
        repaired = fp.with_suffix(".tmp")
        repaired.write_bytes(b"".join(rp.canonical(r) + b"\n" for r in done.values()))
        repaired.replace(fp)  # Remove only duplicate/partial output records on resume.
    report = {"model_tag": tag, "manifest_sha256": manifest["manifest_sha256"],
              "torch": torch.__version__, "transformers": transformers.__version__,
              "runner_sha256": rp.file_hash(Path(__file__)),
              "gpu": torch.cuda.get_device_name(), "status": "started", "historical": []}
    try:
        spec = manifest["models"][tag]
        snapshot = Path("/cache/hub") / ("models--" + spec["model"].replace("/", "--")) / "snapshots" / spec["revision"]
        for name, expected in manifest["source_hashes"][tag]["tokenizer_files"].items():
            if rp.file_hash(snapshot / name) != expected:
                raise ValueError(f"Remote tokenizer differs from preparation: {name}")
        tok = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        # Stream whole shards to ephemeral local disk before safetensors mmap.
        # Direct mmap of this existing Volume was extremely slow in the smoke run.
        from concurrent.futures import ThreadPoolExecutor
        import shutil
        staged = Path("/tmp/opening-ranks-model") / tag / spec["revision"]
        staged.mkdir(parents=True, exist_ok=True)
        def stage_file(src):
            dst = staged / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                return
            tmp = dst.with_suffix(dst.suffix + ".partial")
            with src.open("rb") as reader, tmp.open("wb") as writer:
                shutil.copyfileobj(reader, writer, length=16 * 1024 * 1024)
            tmp.replace(dst)
            if src.suffix == ".safetensors":
                print(f"[{tag}] staged {src.name}", flush=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(stage_file, [f for f in snapshot.iterdir() if f.is_file()]))
        report["stage_seconds"] = time.monotonic() - started
        print(f"[{tag}] local weight staging complete in {report['stage_seconds']:.1f}s", flush=True)
        # No CPU offload or quantization: 32B bf16 plus short prefixes fits H100.
        model = AutoModelForCausalLM.from_pretrained(staged, torch_dtype=torch.bfloat16,
                                                    device_map="cuda", attn_implementation="sdpa",
                                                    local_files_only=True).eval()
        vocab = model.config.vocab_size
        # In this tokenizer version len(tok) scans its vocabulary (~15ms/call).
        tokenizer_size = len(tok)
        texts = [tok.decode([i], skip_special_tokens=False, clean_up_tokenization_spaces=False)
                 if i < tokenizer_size else "" for i in range(vocab)]
        mask_cache = {}
        special_ids = set(tok.all_special_ids)
        def get_masks(pair):
            key = rp.mask_key(pair["ctype"], pair["marker"], pair["state"])
            if key not in mask_cache:
                mask_cache[key] = rp.masks_from_state(texts, special_ids, pair["ctype"], pair["marker"], pair["state"])
            return mask_cache[key]
        report["load_seconds"] = time.monotonic() - started
        print(f"[{tag}] model ready in {report['load_seconds']:.1f}s, vocab={vocab}", flush=True)

        forward_cache = {}
        def forward(ids):
            key = tuple(ids)
            if key in forward_cache:
                return forward_cache[key]
            x = torch.tensor([ids], dtype=torch.long, device="cuda")
            with torch.inference_mode():
                # Last input position predicts the not-yet-supplied target token.
                # Keep the historical full-prefix projection shape. A one-row
                # bf16 output projection changes rounding and can alter tied ranks.
                logits = model(input_ids=x, use_cache=False).logits[0, -1].float()
            result = logits.cpu().numpy()
            del logits, x
            forward_cache[key] = result
            return result

        # Repeated-input check and a diagnostic for the one-row optimization.
        sample = manifest["pairs"][tag][0]
        sample_ids = sample["prompt_ids"]["instruction_only"] + sample["prefix_ids"]
        with torch.inference_mode():
            full = model(input_ids=torch.tensor([sample_ids], device="cuda"), use_cache=False).logits[0, -1].float().cpu().numpy()
            selective = model(input_ids=torch.tensor([sample_ids], device="cuda"), use_cache=False,
                              logits_to_keep=1).logits[0, -1].float().cpu().numpy()
        repeated = forward(sample_ids)
        report["projection"] = "Full short-prefix logits, select final input position"
        report["selective_projection_max_abs_diff"] = float(np.max(np.abs(full - selective)))
        report["indexing_check_max_abs"] = float(np.max(np.abs(full - repeated)))
        if not np.array_equal(full, repeated):
            raise ValueError("Identical forward requests did not reproduce exact logits")

        for check in manifest["historical_checks"][tag]:
            z = forward(check["input_ids"])
            lp = rp.base.log_softmax(z)
            top5 = np.argsort(-z, kind="stable")[:5]
            old_ids = np.asarray(check["top5_ids"])
            old_p = np.asarray(check["top5_p"])
            # fp16 old probabilities can round to zero: exclude those log comparisons.
            nonzero = old_p > 0
            max_log_diff = float(np.max(np.abs(lp[old_ids[nonzero]] - np.log(old_p[nonzero]))))
            entropy = float(-(np.exp(lp) * lp).sum())
            overlap = len(set(top5) & set(old_ids))
            entry = {"trace_id": check["trace_id"], "position": check["position"],
                     "top5_overlap": overlap, "max_top5_logp_diff": max_log_diff,
                     "entropy_abs_diff": abs(entropy - check["entropy"])}
            entry["passed"] = overlap >= 3 and max_log_diff <= 1.0 and entry["entropy_abs_diff"] <= 0.15
            report["historical"].append(entry)
            if not entry["passed"]:
                raise ValueError(f"Historical reproduction failed: {entry}")
        print(f"[{tag}] indexing and all {len(report['historical'])} historical checks passed", flush=True)

        all_pairs = manifest["pairs"][tag]
        smoke_ids = set()
        for ct in rp.TYPES:
            pool = [p for p in all_pairs if p["ctype"] == ct]
            smoke_ids.add(pool[0]["pair_id"])
            smoke_ids.add(max(pool, key=lambda p: len(p["prefix_ids"]))["pair_id"])
        smoke_count = len(smoke_ids - set(done))
        pending = [p for p in all_pairs if p["pair_id"] not in done]
        pending.sort(key=lambda p: p["pair_id"] not in smoke_ids)
        timings, token_rates = [], []
        for n, pair in enumerate(pending):
            elapsed = time.monotonic() - started
            estimate = float(np.median(timings)) if timings else 30.0
            if elapsed + estimate + 60 > max_seconds:
                report["status"] = "budget_stopped"
                break
            masks = get_masks(pair)
            before = time.monotonic()
            zs = {cond: forward(pair["prompt_ids"][cond] + pair["prefix_ids"]) for cond in rp.CONDS}
            stat = rp.base.paired_statistics(zs, masks, texts, pair["observed_ids"])
            row = {"version": rp.VERSION, "manifest_sha256": manifest["manifest_sha256"],
                   "pair_id": pair["pair_id"], "model_tag": tag, "ctype": pair["ctype"],
                   "prompt_id": pair["prompt_id"], "question_id": pair["question_id"],
                   "position": pair["position"], "prefix_ids": pair["prefix_ids"],
                   "marker": pair["marker"], "state": pair["state"], "constraint_active": pair["constraint_active"],
                   "references": pair["references"], **stat}
            with fp.open("ab") as f:
                f.write(rp.canonical(row) + b"\n")
                f.flush()
            done[pair["pair_id"]] = row
            timings.append(time.monotonic() - before)
            tokens = sum(len(pair["prompt_ids"][c]) + len(pair["prefix_ids"]) for c in rp.CONDS)
            token_rates.append(timings[-1] / tokens)
            if n + 1 == smoke_count:
                remaining_tokens = sum(sum(len(p["prompt_ids"][c]) + len(p["prefix_ids"]) for c in rp.CONDS) for p in pending[n+1:])
                # Includes long first-failure prefixes; allow 120s for new mask sets.
                projected = remaining_tokens * max(token_rates) + 120
                report["smoke"] = {"pairs": smoke_count, "median_seconds_per_pair": float(np.median(timings)),
                                    "projected_remaining_seconds": projected,
                                    "max_smoke_input_tokens": max(len(p["prompt_ids"]["instruction_only"])+len(p["prefix_ids"]) for p in pending[:n+1])}
                print(f"[{tag}] smoke: {report['smoke']}", flush=True)
                if time.monotonic() - started + projected + 60 > max_seconds:
                    report["status"] = "projection_exceeds_budget"
                    break
            if (n + 1) % 25 == 0 or n + 1 == smoke_count:
                results_volume.commit()
                print(f"[{tag}] {len(done)}/{len(all_pairs)} unique paired prefixes scored", flush=True)
        else:
            report["status"] = "complete"
        report["pairs_done"] = len(done)
        report["pairs_expected"] = len(all_pairs)
        report["unique_forward_passes"] = len(forward_cache)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[{tag}] {report['error']}", flush=True)
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        (out / "report.json").write_text(json.dumps(report, indent=2))
        results_volume.commit()
        # Models are sequential; release CPU/GPU storage before a reused worker.
        if "model" in locals():
            del model
        if "forward_cache" in locals():
            forward_cache.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return {"report": report, "results": list(done.values())}


@app.local_entrypoint()
def main(run: str = "opening_suite_v1"):
    import sys
    sys.path.insert(0, str(HERE))
    import suite_rank_probe as rp
    root = HERE / "data/rank_runs" / run
    manifest = rp.load_manifest(root / "manifest.json")
    if rp.file_hash(HERE / "suite_rank_probe.py") != manifest["scorer_sha256"]:
        raise ValueError("Scorer changed after preparation; create a new manifest/run")
    ledger_path = root / "budget_ledger.json"
    ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {
        "manifest_sha256": manifest["manifest_sha256"], "max_gpu_seconds": 3600,
        "initial_budget_usd": 10, "ceiling_usd": 10, "conservative_compute_usd_per_hour": 6,
        "attempts": []}
    if ledger["manifest_sha256"] != manifest["manifest_sha256"]:
        raise ValueError("Budget ledger belongs to a different manifest")

    def save_ledger():
        tmp = ledger_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(ledger, indent=2))
        tmp.replace(ledger_path)

    for tag in rp.MODELS:
        report_path = root / f"{tag}_report.json"
        if report_path.exists() and json.loads(report_path.read_text())["status"] == "complete":
            print(f"{tag}: already complete; no worker launched")
            continue
        charged = sum(a.get("accounted_seconds", a["reserved_seconds"]) for a in ledger["attempts"])
        available = min(1800, ledger["max_gpu_seconds"] - charged,
                        (ledger["initial_budget_usd"] / 6 * 3600) - charged)
        if available < 300:
            raise RuntimeError("No remaining experiment runtime budget; partial results retained")
        attempt = {"model_tag": tag, "reserved_seconds": available,
                   "started_at_unix": time.time(), "status": "reserved"}
        ledger["attempts"].append(attempt)
        save_ledger()  # An uncertain/terminated call retains the FULL reservation.
        print(f"Launching {tag}: reserved {available / 3600:.2f} GPU-hours; sequential execution", flush=True)
        # Enforce the reservation even if weight loading or a forward hangs.
        result = rescore.with_options(timeout=int(available)).remote(manifest, tag, available - 60)
        report = result["report"]
        attempt.update(status=report["status"], accounted_seconds=min(available, report["elapsed_seconds"] + 60))
        ledger["estimated_compute_usd_upper"] = sum(a.get("accounted_seconds", a["reserved_seconds"])
                                                     for a in ledger["attempts"]) / 3600 * 6
        save_ledger()
        report_path.write_text(json.dumps(report, indent=2))
        with (root / f"{tag}_pairs.jsonl").open("wb") as f:
            for row in result["results"]:
                f.write(rp.canonical(row) + b"\n")
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "complete":
            raise RuntimeError(f"Stopped on {tag}: {report['status']}; inspect report before any retry")
    print(f"Finished. Conservative compute estimate: ${ledger.get('estimated_compute_usd_upper', 0):.2f}")
