"""Test OpenRouter concurrency limits for a given model.

Fires batches of trivial requests at increasing concurrency levels and
reports throughput, errors, and wall-clock time.

Usage:
    uv run python scripts/test_concurrency.py --model qwen3-8b
    uv run python scripts/test_concurrency.py --model qwen3-8b --levels 5,10,20,30,50 --requests-per-level 20
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path so `lib` is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import get_model_by_name, load_config
from lib.providers.openrouter import async_complete

TRIVIAL_PROMPT = [{"role": "user", "content": "What is 2+2? Reply with just the number."}]


async def run_batch(model_id: str, concurrency: int, n_requests: int) -> dict:
    """Fire n_requests with given concurrency, return stats."""
    sem = asyncio.Semaphore(concurrency)
    results = {"success": 0, "error": 0, "errors": []}

    async def _one():
        async with sem:
            try:
                await async_complete(model_id, TRIVIAL_PROMPT, max_tokens=50, temperature=0.0)
                results["success"] += 1
            except Exception as e:
                results["error"] += 1
                err_str = str(e)[:120]
                if err_str not in results["errors"]:
                    results["errors"].append(err_str)

    t0 = time.monotonic()
    await asyncio.gather(*[_one() for _ in range(n_requests)])
    elapsed = time.monotonic() - t0

    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "success": results["success"],
        "errors": results["error"],
        "elapsed": elapsed,
        "rps": n_requests / elapsed if elapsed > 0 else 0,
        "error_msgs": results["errors"],
    }


async def main(model_id: str, levels: list[int], requests_per_level: int):
    print(f"Model: {model_id}")
    print(f"Requests per level: {requests_per_level}")
    print(f"Levels: {levels}")
    print()
    print(f"{'Concurrency':>12} {'Success':>8} {'Errors':>8} {'Time (s)':>10} {'Req/s':>8}")
    print("-" * 52)

    for level in levels:
        stats = await run_batch(model_id, level, requests_per_level)
        print(
            f"{stats['concurrency']:>12} "
            f"{stats['success']:>8} "
            f"{stats['errors']:>8} "
            f"{stats['elapsed']:>10.1f} "
            f"{stats['rps']:>8.1f}"
        )
        if stats["error_msgs"]:
            for msg in stats["error_msgs"]:
                print(f"    ^ {msg}")

    print()
    print("Use the highest concurrency level with 0 errors for your config.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenRouter concurrency limits")
    parser.add_argument("--model", type=str, default="qwen3-8b", help="Model short name")
    parser.add_argument("--levels", type=str, default="5,10,20,30,50",
                        help="Comma-separated concurrency levels to test")
    parser.add_argument("--requests-per-level", type=int, default=20,
                        help="Number of requests per concurrency level")
    args = parser.parse_args()

    config = load_config()
    model = get_model_by_name(config, args.model)
    levels = [int(x) for x in args.levels.split(",")]

    asyncio.run(main(model.model_id, levels, args.requests_per_level))
