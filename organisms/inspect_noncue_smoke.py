"""Inspect non-cue arm smoke test: rewrite quality & training pairs."""
import pandas as pd
from IPython.display import display, HTML
pd.set_option('display.max_colwidth', 200)

# --- Load ---
ARTIFACT = "counterfactual-simulation-training/artifacts/"
smoke_file = [f for f in __import__('pathlib').Path(ARTIFACT).glob("*noncue_smoke*train*.csv")][0]
df = pd.read_csv(smoke_file)
print(f"Loaded {len(df)} rows from {smoke_file.name}\n")

# --- Overview ---
print("="*60)
print("OVERVIEW")
print("="*60)
print(f"Model accuracy:        {df['pred_acc_0'].mean():.2f}")
print(f"Sim accuracy (xye):    {df['simulator_acc_0'].mean():.2f}")
print(f"Rewrite attempted:     {df['attempted_rewrite'].sum()} / {len(df)}")
print(f"Rewrite success:       {(df['rewrite_success_round'] > 0).sum()} / {df['attempted_rewrite'].sum()}")
print(f"Valid positives:       {(df['positive_reasoning'] != '').sum()} / {len(df)}")
print(f"Valid negatives:       {(df['negative_reasoning'] != '').sum()} / {len(df)}")
print()

# Source breakdown
print("Positive sources:", df['positive_source'].value_counts().to_dict())
print("Negative sources:", df['negative_source'].value_counts().to_dict())
print()

# --- Per-point detail ---
print("="*60)
print("PER-POINT DETAIL")
print("="*60)
for i, row in df.iterrows():
    print(f"\n{'─'*60}")
    print(f"POINT {i} | id={row['id']} | subject={row.get('subject','?')}")
    print(f"  Model correct: {bool(row['pred_acc_0'])} | Sim correct (xye): {bool(row['simulator_acc_0'])}")
    print(f"  Rewrite attempted: {bool(row['attempted_rewrite'])} | Success round: {row['rewrite_success_round']}")
    print(f"  Positive source: {row['positive_source']} | Negative source: {row['negative_source']}")

    # Show the question (truncated)
    q = str(row['original_question'])[:300]
    print(f"\n  QUESTION: {q}...")
    print(f"  MODEL ANSWER: {row['original_model_answer_0']} (correct={row['formatted_answer']})")
    print(f"  CF ANSWER:    {row['counterfactual_model_answer_0']}")

    # Original CoT vs Rewritten CoT (truncated)
    orig_cot = str(row.get('original_model_cot_0', ''))[:400]
    rewritten = str(row.get('rewritten_reasoning', ''))[:400]
    print(f"\n  ORIGINAL COT (first 400 chars):\n    {orig_cot}")
    if rewritten and rewritten != 'nan':
        print(f"\n  REWRITTEN COT (first 400 chars):\n    {rewritten}")
        print(f"\n  REWRITE SIM ACC: {row['rewrite_simulator_acc']} (want False for success)")

    # Positive / Negative
    pos = str(row.get('positive_reasoning', ''))[:300]
    neg = str(row.get('negative_reasoning', ''))[:300]
    if pos and pos != 'nan':
        print(f"\n  POSITIVE (first 300 chars): {pos}")
    if neg and neg != 'nan':
        print(f"\n  NEGATIVE (first 300 chars): {neg}")

print(f"\n{'='*60}")
print("KEY: positive = uninformative CoT (goal), negative = informative CoT (original)")
print("Rewrite success = sim FAILS on rewritten CoT (can't predict CF answer)")
