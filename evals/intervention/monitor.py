"""Monitor module for intervention evaluations.

The monitor determines whether a model's reasoning was influenced by the
intervention (hint/framing). Reuses the judge client infrastructure from
lib/judge.py but with different prompts and output parsing.
"""

import re
from dataclasses import dataclass
from typing import Optional

from lib.judge import JudgeConfig, get_async_judge_client


@dataclass
class MonitorResult:
    """Result from a monitor evaluation."""

    output_letter: Optional[str]  # A/B/C/D for math, X/Y for scruples
    z: Optional[int]  # 1 = influenced, 0 = not influenced
    explanation: str
    raw: str


async def async_monitor(
    config: JudgeConfig,
    messages: list[dict],
) -> MonitorResult:
    """Run the monitor on a single sample.

    Args:
        config: Monitor model configuration (reuses JudgeConfig).
        messages: Pre-formatted monitor prompt messages.

    Returns:
        MonitorResult with output letter, z value, and explanation.
    """
    client = get_async_judge_client(config)

    # Build API kwargs
    # gpt-5+, o-series, and other newer OpenAI models require max_completion_tokens
    uses_completion_tokens = (
        config.model_id.startswith("o")
        or config.model_id.startswith("gpt-5")
    )
    token_param = "max_completion_tokens" if uses_completion_tokens else "max_tokens"
    kwargs = {
        "model": config.model_id,
        "messages": messages,
        token_param: config.max_tokens,
    }
    if config.reasoning_effort and config.model_id.startswith("o"):
        kwargs["reasoning_effort"] = config.reasoning_effort
    else:
        kwargs["temperature"] = config.temperature

    try:
        response = await client.chat.completions.create(**kwargs)
        raw = response.choices[0].message.content or ""
        return MonitorResult(
            output_letter=None,
            z=None,
            explanation=raw,
            raw=raw,
        )
    except Exception as e:
        return MonitorResult(
            output_letter=None,
            z=None,
            explanation=f"Monitor error: {e}",
            raw="",
        )
