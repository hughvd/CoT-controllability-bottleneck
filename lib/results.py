"""Result storage and loading utilities.

Two storage modes:
1. Legacy: Timestamped JSON files per stage (save_results / load_latest_results)
2. Granular cache: Key-value JSON per stage, keyed by completion ID.
   Never redoes an API call — load cache, skip existing keys, save after each new result.
"""

import asyncio
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


# ── Legacy API (timestamped files) ─────────────────────────────────

def get_results_dir(eval_name: str, model_name: str, stage: str) -> Path:
    """Get the results directory for a specific eval/model/stage combination."""
    path = RESULTS_DIR / eval_name / model_name / stage
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_results(
    eval_name: str,
    model_name: str,
    stage: str,
    data: dict[str, Any],
    filename: Optional[str] = None,
) -> Path:
    """Save results to a timestamped JSON file."""
    results_dir = get_results_dir(eval_name, model_name, stage)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}.json"
    path = results_dir / filename
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def load_latest_results(
    eval_name: str,
    model_name: str,
    stage: str,
) -> Optional[dict[str, Any]]:
    """Load the most recent results file for a given eval/model/stage."""
    results_dir = get_results_dir(eval_name, model_name, stage)
    json_files = sorted(results_dir.glob("*.json"))
    if not json_files:
        return None
    with open(json_files[-1]) as f:
        return json.load(f)


def results_exist(eval_name: str, model_name: str, stage: str) -> bool:
    """Check if any results exist for a given eval/model/stage."""
    results_dir = RESULTS_DIR / eval_name / model_name / stage
    if not results_dir.exists():
        return False
    return bool(list(results_dir.glob("*.json")))


# ── Granular cache API (key-value per stage) ────────────────────────

def _cache_path(eval_name: str, model_name: str, stage: str) -> Path:
    """Path to the granular cache file for a stage."""
    model_dir = RESULTS_DIR / eval_name / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / f"{stage}.json"


def load_cache(eval_name: str, model_name: str, stage: str) -> dict[str, Any]:
    """Load the granular cache for a stage. Returns empty dict if none exists."""
    path = _cache_path(eval_name, model_name, stage)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(eval_name: str, model_name: str, stage: str, cache: dict[str, Any]) -> Path:
    """Save the granular cache atomically (write to temp, then rename)."""
    path = _cache_path(eval_name, model_name, stage)
    # Atomic write: write to temp file in same directory, then rename
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, suffix=".tmp", delete=False
    ) as f:
        json.dump(cache, f, indent=2, default=str)
        tmp_path = Path(f.name)
    tmp_path.rename(path)
    return path


def cache_get(cache: dict[str, Any], key: str) -> Optional[Any]:
    """Get a value from cache, or None if not present."""
    return cache.get(key)


def cache_set(
    cache: dict[str, Any],
    key: str,
    value: Any,
    eval_name: str,
    model_name: str,
    stage: str,
    auto_save: bool = True,
) -> None:
    """Set a value in cache and optionally persist to disk."""
    cache[key] = value
    if auto_save:
        save_cache(eval_name, model_name, stage, cache)


# ── Async cache (for concurrent pipelines) ────────────────────────

class AsyncCache:
    """Async-safe cache wrapper with periodic disk saves.

    Usage:
        cache = AsyncCache("eval", "model", "stage")
        await cache.load()
        # Build todo list using `key not in cache` (safe before gather)
        await asyncio.gather(*[worker(item) for item in todo])
        # Each worker calls `await cache.set(key, value)`
        await cache.flush()
    """

    def __init__(self, eval_name: str, model_name: str, stage: str,
                 save_every: int = 25):
        self.eval_name = eval_name
        self.model_name = model_name
        self.stage = stage
        self.save_every = save_every
        self._lock = asyncio.Lock()
        self._data: dict[str, Any] = {}
        self._unsaved = 0

    async def load(self):
        """Load existing cache from disk. Call once before use."""
        self._data = load_cache(self.eval_name, self.model_name, self.stage)

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str) -> Any | None:
        return self._data.get(key)

    @property
    def data(self) -> dict[str, Any]:
        """Direct access to underlying dict (for read-only iteration after gather)."""
        return self._data

    async def set(self, key: str, value: Any):
        """Set a value, periodically flushing to disk."""
        async with self._lock:
            self._data[key] = value
            self._unsaved += 1
            if self._unsaved >= self.save_every:
                save_cache(self.eval_name, self.model_name, self.stage, self._data)
                self._unsaved = 0

    async def flush(self):
        """Force save to disk. Call at end of each stage."""
        async with self._lock:
            if self._unsaved > 0:
                save_cache(self.eval_name, self.model_name, self.stage, self._data)
                self._unsaved = 0
