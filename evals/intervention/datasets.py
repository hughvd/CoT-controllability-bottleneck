"""Dataset loading for intervention evaluations.

Auto-downloads from HuggingFace on first use. Cached locally after.
"""

from typing import Optional


def load_gpqa(n_problems: Optional[int] = None) -> list[dict]:
    """Load GPQA Diamond split from HuggingFace.

    Returns list of dicts with keys:
        - id: unique identifier
        - question: full question text with answer choices
        - answer: correct answer letter (A/B/C/D)
        - choices: dict mapping A/B/C/D to choice text
    """
    import random

    from datasets import load_dataset

    ds = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")

    problems = []
    for i, row in enumerate(ds):
        question_text = row.get("Question", row.get("question", ""))
        correct_answer = row.get("Correct Answer", row.get("correct_answer", ""))

        # Build choices from Correct Answer + Incorrect Answer 1/2/3
        all_choices = [correct_answer]
        for j in range(1, 4):
            inc = row.get(f"Incorrect Answer {j}", "")
            if inc:
                all_choices.append(inc)

        # Shuffle with fixed seed per problem for reproducibility
        rng = random.Random(i)
        rng.shuffle(all_choices)

        # Assign to letters
        letters = ["A", "B", "C", "D"]
        choices = {}
        answer_letter = "A"
        for k, choice_text in enumerate(all_choices[:4]):
            choices[letters[k]] = choice_text
            if choice_text == correct_answer:
                answer_letter = letters[k]

        # Format question with choices
        formatted_choices = "\n".join(f"({letter}) {text}" for letter, text in choices.items())
        question_text = f"{question_text}\n\n{formatted_choices}"

        problems.append({
            "id": str(i),
            "question": question_text,
            "answer": answer_letter,
            "choices": choices,
        })

    if n_problems is not None:
        problems = problems[:n_problems]

    return problems


def load_daft_math(n_problems: Optional[int] = None) -> list[dict]:
    """Load DAFT Math dataset.

    Returns list of dicts with keys:
        - id: unique identifier
        - question: problem text
        - answer: correct integer answer
    """
    from datasets import load_dataset

    ds = load_dataset("metr-evals/daft-math", split="train")

    problems = []
    for i, row in enumerate(ds):
        # Use Integer Answer Variant (integer answers for easy parsing)
        question = row.get("Integer Answer Variant Question", row.get("Original Question", ""))
        answer = row.get("Integer Variant Answer", row.get("Original Answer", ""))
        if not question or not answer:
            continue
        problems.append({
            "id": str(i),
            "question": question,
            "answer": str(answer),
        })

    if n_problems is not None:
        problems = problems[:n_problems]

    return problems


def load_aime(n_problems: Optional[int] = None) -> list[dict]:
    """Load AIME problems from HuggingFace (qq8933/AIME_1983_2024).

    Returns list of dicts with keys:
        - id: unique identifier
        - question: problem text
        - answer: correct integer answer (0-999)
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("qq8933/AIME_1983_2024", split="train")
    except Exception as e:
        print(f"Warning: Could not load AIME dataset: {e}")
        return []

    problems = []
    for i, row in enumerate(ds):
        question = row.get("Question", row.get("question", ""))
        answer = row.get("Answer", row.get("answer", ""))
        if not question or not answer:
            continue
        problems.append({
            "id": str(i),
            "question": question,
            "answer": str(answer),
        })

    if n_problems is not None:
        problems = problems[:n_problems]

    return problems


def load_scruples(n_problems: Optional[int] = None) -> list[dict]:
    """Load Scruples AITA anecdotes, filtered to posts where author is wrong.

    Uses metaeval/scruples mirror on HuggingFace.

    Returns list of dicts with keys:
        - id: unique identifier
        - title: post title
        - text: post body
        - label: ground truth (should be "WRONG" for filtered set)
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("metaeval/scruples", split="train")
    except Exception as e:
        print(f"Warning: Could not load Scruples dataset: {e}")
        return []

    problems = []
    for i, row in enumerate(ds):
        # Filter to posts where author is in the wrong
        label = row.get("binarized_label", row.get("label", ""))
        if str(label) != "WRONG":
            continue

        title = row.get("title", "")
        text = row.get("text", "")
        if not text:
            continue

        problems.append({
            "id": str(i),
            "title": title,
            "text": text,
            "label": "WRONG",
            "answer": "A",  # Correct judgment: author IS wrong
        })

    if n_problems is not None:
        problems = problems[:n_problems]

    return problems


def load_wmdp(n_problems: Optional[int] = None) -> list[dict]:
    """Load WMDP (Weapons of Mass Destruction Proxy) multiple choice questions.

    Returns list of dicts with keys:
        - id: unique identifier
        - question: question text with choices
        - answer: correct answer index (0-3)
        - choices: list of choice texts
    """
    from datasets import load_dataset

    try:
        ds = load_dataset("cais/wmdp", "wmdp-cyber", split="test")
    except Exception:
        print("Warning: Could not load WMDP dataset. Using empty list.")
        return []

    problems = []
    for i, row in enumerate(ds):
        question = row.get("question", "")
        choices = row.get("choices", [])
        answer_idx = row.get("answer", 0)

        # Format with letter choices
        formatted_choices = "\n".join(
            f"({chr(65 + j)}) {choice}" for j, choice in enumerate(choices)
        )
        full_question = f"{question}\n\n{formatted_choices}"

        problems.append({
            "id": str(i),
            "question": full_question,
            "answer": chr(65 + answer_idx),  # Convert index to letter
            "choices": choices,
        })

    if n_problems is not None:
        problems = problems[:n_problems]

    return problems


# Registry for easy lookup
DATASET_LOADERS = {
    "gpqa": load_gpqa,
    "gpqa_encoded": load_gpqa,  # Same dataset, different hints
    "daft": load_daft_math,
    "daft_math": load_daft_math,
    "aime": load_aime,
    "scruples": load_scruples,
    "wmdp": load_wmdp,
}
