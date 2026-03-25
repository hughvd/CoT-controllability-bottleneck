#!/usr/bin/env python3
"""
generate_cue_positives.py — Generate cue-following CoTs from Qwen3-32B via vLLM.

For each row in the input cue CSV, queries the model with a system prompt that
instructs it to follow the embedded cue. Produces a new CSV with:
  - positive_reasoning / positive_answer = generated cue-following CoT
  - negative_reasoning / negative_answer = original model CoT (for not_influenced rows)
  - original_question = counterfactual_question_0 (the cue question)

Usage:
    python generate_cue_positives.py \
        --input_csv artifacts/dataset_..._train.csv \
        --output_csv artifacts/generated_cue_following_qwen3-32b_n1000.csv \
        --vllm_base_url http://localhost:8000/v1 \
        --model_name qwen3-32b-base \
        --max_tokens 2048
"""

import argparse
import asyncio
from collections import defaultdict

import numpy as np
import openai
import pandas as pd
from tqdm.asyncio import tqdm_asyncio

from globals import bias_description_dict, cue_follow_generation_prompt, mc_task_instructions
from utils import extract_between_tags


CUE_KEYWORDS = [
    "cue", "hint", "suggest", "note", "comment", "mark", "key",
    "professor", "authority", "expert", "endorses",
]


def mentions_cue(reasoning: str, cue_type: str) -> bool:
    """Soft keyword filter — returns True if reasoning mentions any cue-related term."""
    if not reasoning:
        return False
    text = reasoning.lower()
    return any(kw in text for kw in CUE_KEYWORDS)


def parse_response(content: str) -> tuple[str, str]:
    """Extract <think> reasoning and final answer from model response."""
    import re
    reasoning = extract_between_tags(content, None, opener="<think>", closer="</think>")
    # Answer is everything after </think>
    think_end = content.rfind("</think>")
    if think_end != -1:
        answer_text = content[think_end + len("</think>"):].strip()
    else:
        answer_text = content.strip()
    # Look for answer patterns like "A", "(A)", "**A**", "Answer: A", "answer is A"
    # Try structured patterns first, fall back to last standalone letter
    patterns = [
        r'(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-H])\)?',  # "answer is A", "Answer: (B)"
        r'\*\*([A-H])\*\*',                                       # **A**
        r'^\(?([A-H])\)?[\.\s,]',                                 # starts with "A." or "(A)"
        r'\b([A-H])\b\s*$',                                       # ends with standalone letter
        r'\b([A-H])\b',                                            # any standalone letter (last resort)
    ]
    for pattern in patterns:
        matches = re.findall(pattern, answer_text, re.IGNORECASE | re.MULTILINE)
        if matches:
            return reasoning, matches[-1].upper()
    return reasoning, ""


async def generate_one(
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    max_retries: int = 3,
) -> str | None:
    """Query vLLM for a single generation with retries."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"  Failed after {max_retries} attempts: {e}")
                    return None


async def process_row(
    idx: int,
    row: pd.Series,
    client: openai.AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    max_tokens: int,
    temperature: float,
) -> dict | None:
    """Generate a cue-following CoT for one row and return result dict."""
    cue_type = row["counterfactual_type"]
    cue_question = row["counterfactual_question_0"]
    cue_target = str(row["cue_points_to"]).strip().upper()

    # Build system prompt with bias description
    bias_desc = bias_description_dict.get(cue_type, f"A cue of type '{cue_type}'.")
    system_prompt = cue_follow_generation_prompt.format(bias_description=bias_desc)

    # User prompt is the task instructions + the cue question
    user_prompt = f"{mc_task_instructions}\n\n{cue_question}"

    content = await generate_one(
        client, semaphore, model_name, system_prompt, user_prompt,
        max_tokens, temperature,
    )
    if content is None:
        return None

    reasoning, answer = parse_response(content)

    # Filter: answer must match cue target AND reasoning must mention the cue
    if answer != cue_target:
        return {"idx": idx, "reject_reason": "wrong_answer"}
    if not mentions_cue(reasoning, cue_type):
        return {"idx": idx, "reject_reason": "no_cue_mention"}

    # Build result
    result = {"idx": idx, "reject_reason": None}
    result["positive_reasoning"] = reasoning
    result["positive_answer"] = answer
    result["positive_example_score"] = 1.0

    # Contrastive pair: use original CoT as negative for not_influenced rows
    influence_type = row.get("cue_influence_type", "")
    if influence_type == "not_influenced":
        result["negative_reasoning"] = row.get("counterfactual_model_cot_round0", "")
        result["negative_answer"] = str(row.get("original_model_answer_0", "")).strip()
        result["negative_example_score"] = 0.0
    else:
        result["negative_reasoning"] = np.nan
        result["negative_answer"] = np.nan
        result["negative_example_score"] = np.nan

    return result


async def main(args):
    print(f"Loading input CSV: {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    print(f"  Rows: {len(df)}, Cue types: {df['counterfactual_type'].nunique()}")
    print(f"  Influence distribution:\n{df['cue_influence_type'].value_counts().to_string()}")
    print()

    client = openai.AsyncOpenAI(base_url=args.vllm_base_url, api_key="dummy")
    semaphore = asyncio.Semaphore(args.concurrency)

    # Launch all generation tasks
    tasks = []
    for idx, row in df.iterrows():
        tasks.append(
            process_row(idx, row, client, semaphore, args.model_name, args.max_tokens, args.temperature)
        )

    results = await tqdm_asyncio.gather(*tasks, desc="Generating cue-following CoTs")

    # Collect results and stats
    reject_counts = defaultdict(int)
    success_count = 0
    paired_count = 0
    unpaired_count = 0
    cue_type_counts = defaultdict(int)
    accepted_indices = set()

    for r in results:
        if r is None:
            reject_counts["api_error"] += 1
            continue
        if r["reject_reason"] is not None:
            reject_counts[r["reject_reason"]] += 1
            continue
        idx = r["idx"]
        accepted_indices.add(idx)
        success_count += 1
        cue_type_counts[df.loc[idx, "counterfactual_type"]] += 1
        if pd.notna(r.get("negative_example_score")):
            paired_count += 1
        else:
            unpaired_count += 1

    # Build output DataFrame — keep only accepted rows
    out_df = df.loc[list(accepted_indices)].copy()

    # Overwrite columns with generated data
    for idx in accepted_indices:
        r = next(x for x in results if x is not None and x.get("idx") == idx and x["reject_reason"] is None)
        out_df.loc[idx, "positive_reasoning"] = r["positive_reasoning"]
        out_df.loc[idx, "positive_answer"] = r["positive_answer"]
        out_df.loc[idx, "positive_example_score"] = r["positive_example_score"]
        out_df.loc[idx, "negative_reasoning"] = r["negative_reasoning"]
        out_df.loc[idx, "negative_answer"] = r["negative_answer"]
        out_df.loc[idx, "negative_example_score"] = r["negative_example_score"]
        # Critical: training reads original_question as model input; for cue arm this must be the cue question
        out_df.loc[idx, "original_question"] = df.loc[idx, "counterfactual_question_0"]

    out_df.to_csv(args.output_csv, index=False)

    # Print summary
    total = len(df)
    print(f"\n{'='*60}")
    print(f"Generation Summary")
    print(f"{'='*60}")
    print(f"Total rows:        {total}")
    print(f"Accepted:          {success_count} ({100*success_count/total:.1f}%)")
    print(f"  Paired (w/ neg): {paired_count}")
    print(f"  Unpaired:        {unpaired_count}")
    print(f"Rejected:          {sum(reject_counts.values())}")
    for reason, count in sorted(reject_counts.items()):
        print(f"  {reason}: {count}")
    print(f"\nPer cue-type counts:")
    for ct, count in sorted(cue_type_counts.items()):
        print(f"  {ct}: {count}")
    print(f"\nOutput written to: {args.output_csv}")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cue-following CoTs via vLLM")
    parser.add_argument("--input_csv", required=True, help="Input cue CSV path")
    parser.add_argument("--output_csv", required=True, help="Output CSV path")
    parser.add_argument("--vllm_base_url", default="http://localhost:8000/v1", help="vLLM base URL")
    parser.add_argument("--model_name", default="qwen3-32b-base", help="Model name served by vLLM")
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--concurrency", type=int, default=20, help="Max concurrent requests")
    args = parser.parse_args()

    asyncio.run(main(args))
