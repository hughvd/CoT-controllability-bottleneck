"""Generate F2 single-category transfer configs: per model, one run per
category (train 1 category, eval the full universe). Recipe = r64 (the v33
winner). Training data is MODEL-INDEPENDENT (same CoTs), so each category's CSV
is shared across models and generated once.
"""
import yaml
from pathlib import Path

TEMPLATE = yaml.safe_load(open("sft/configs/v33_diverse.yaml"))

MODELS = {  # short_key -> base_model (Phase A; gpt-oss/olmo added after integration)
    "qwen3_8b":  "Qwen/Qwen3-8B",
    "qwen3_14b": "Qwen/Qwen3-14B",
    "qwen3_32b": "Qwen/Qwen3-32B",
}

# Fixed train sets, ~2-5 types each to keep run size comparable across categories.
CATS = {
    "transforms": dict(
        families=["uppercase_thinking","alternating_case","prefix_sentences",
                  "start_with","exclaim_all_sentences"],  # = v32 transforms_within
        languages=[], suppression=[],
        held_out_families=TEMPLATE["data"]["held_out_families"]),
    "encoding": dict(
        families=["pig_latin","leetspeak"], languages=[], suppression=[],
        held_out_families=["disemvowel","reverse_each_word","rot13"]),
    "language": dict(
        families=[], languages=["spanish","german","chinese","russian","hebrew"],
        suppression=[], held_out_families=[]),
    "suppression": dict(
        families=[], languages=[], suppression=["omit_word","omit_multiple_words"],
        held_out_families=[]),
}

for model_key, base in MODELS.items():
    for cat, sets in CATS.items():
        cfg = yaml.safe_load(open("sft/configs/v33_diverse.yaml"))
        cfg["version"] = f"f2_{model_key}_{cat}"
        cfg["base_model"] = base
        d = cfg["data"]
        d["train_csv"] = f"data/training_f2_{cat}.csv"   # shared across models
        d["families"] = sets["families"]
        d["languages"] = sets["languages"]
        d["suppression"] = sets["suppression"]
        d["held_out_families"] = sets["held_out_families"]
        d["held_out_languages"] = ([l for l in TEMPLATE["data"]["held_out_languages"]]
                                   if cat != "language" else TEMPLATE["data"]["held_out_languages"])
        d["held_out_suppression"] = (["ignore_question"] if cat == "suppression"
                                     else TEMPLATE["data"]["held_out_suppression"])
        cfg["lora"]["rank"] = 64
        cfg["lora"]["alpha"] = 128
        out = Path(f"sft/configs/f2_{model_key}_{cat}.yaml")
        with open(out, "w") as f:
            f.write(f"# F2 single-category transfer: train {cat} only, eval_all.\n"
                    f"# {model_key} ({base}), r64. Shared data/training_f2_{cat}.csv.\n")
            yaml.safe_dump(cfg, f, sort_keys=False)
print("wrote", len(MODELS)*len(CATS), "configs")
