"""OpenRouter API client using OpenAI-compatible interface.

OpenRouter provides a unified API for open-source models at
https://openrouter.ai/api/v1 with an OpenAI-compatible schema.
"""

import os
import re
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

load_dotenv()

# Models that need explicit reasoning.max_tokens to return thinking in the
# message.reasoning field.  Without this, they return all reasoning in content.
# Most models (Qwen3, DeepSeek, kimi, qwq) return reasoning automatically.
_NEEDS_REASONING_PARAM = {
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "qwen/qwen3-max-thinking",
}


@dataclass
class ModelResponse:
    """Parsed response from a reasoning model."""

    thinking: Optional[str]  # Content inside <think>...</think> tags
    answer: str  # Everything outside thinking tags
    raw: str  # Full raw response text
    finish_reason: str
    usage: dict  # token counts


def get_client() -> OpenAI:
    """Create an OpenRouter client using OpenAI SDK.

    OPENROUTER_BASE_URL overrides the endpoint (e.g. a local vLLM server's
    OpenAI-compatible API for pod-hosted checkpoints).
    """
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not set in environment")
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=_client_timeout(),
    )


def _client_timeout() -> float:
    """HTTP request timeout (seconds). Default 1800 — long-reasoning subjects
    (e.g. OLMo-3 Think locally) can generate for many minutes per request; the
    old 120s cap silently timed out ~50-70% of gpqa/wmdp generations. Override
    with SUBJECT_HTTP_TIMEOUT for fast API models where a real hang should surface
    sooner (retries re-run timed-out records on relaunch either way)."""
    return float(os.environ.get("SUBJECT_HTTP_TIMEOUT", "1800"))


_async_client: AsyncOpenAI | None = None


def get_async_client() -> AsyncOpenAI:
    """Create or return a cached async OpenRouter client."""
    global _async_client
    if _async_client is None:
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in environment")
        _async_client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=_client_timeout(),
        )
    return _async_client


def _parse_thinking(text: str) -> tuple[Optional[str], str]:
    """Extract <think>...</think> blocks from response text.

    Returns (thinking_content, remaining_text).
    """
    pattern = r"<think>(.*?)</think>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        thinking = match.group(1).strip()
        remaining = re.sub(pattern, "", text, flags=re.DOTALL).strip()
        return thinking, remaining
    # Chat templates that force-open the think block (e.g. OLMo-3 Think served
    # without a vLLM reasoning parser) emit only a closing </think> in content:
    # reasoning before it, answer after.
    if "</think>" in text:
        pre, _, post = text.partition("</think>")
        return pre.replace("<think>", "").strip(), post.strip()
    return None, text


def complete(
    model_id: str,
    messages: list[dict],
    *,
    temperature: float = 0.6,
    max_tokens: int = 8192,
    suppress_thinking: bool = False,
    cot_suppression_method: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
) -> ModelResponse:
    """Send a completion request to OpenRouter.

    Args:
        model_id: OpenRouter model ID (e.g. "qwen/qwen3-8b").
        messages: Chat messages in OpenAI format.
        temperature: Sampling temperature.
        max_tokens: Maximum output tokens.
        suppress_thinking: If True, apply CoT suppression for this request.
        cot_suppression_method: "no_think" or "reasoning_complete".

    Returns:
        Parsed ModelResponse with thinking and answer separated.
    """
    client = get_client()

    # Apply CoT suppression
    if suppress_thinking and cot_suppression_method:
        messages = _apply_cot_suppression(messages, cot_suppression_method)
        # For suppressed thinking, use short max_tokens
        max_tokens = min(max_tokens, 100)

    kwargs = dict(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if model_id in _NEEDS_REASONING_PARAM:
        kwargs["extra_body"] = {"reasoning": {"max_tokens": max_tokens}}
    elif reasoning_effort:
        kwargs["extra_body"] = {"reasoning": {"effort": reasoning_effort}}

    response = client.chat.completions.create(**kwargs)

    return _parse_response(response)


def _parse_response(response) -> ModelResponse:
    """Parse an API response into a ModelResponse.

    Checks three sources for thinking/reasoning content:
    1. OpenRouter's `message.reasoning` field (Qwen3, etc.)
    2. `<think>...</think>` tags in message content (DeepSeek R1, etc.)
    3. Falls back to None if neither is present.
    """
    choice = response.choices[0]
    raw_text = choice.message.content or ""

    # Source 1: OpenRouter returns reasoning in a separate field
    reasoning_field = getattr(choice.message, "reasoning", None)

    # Source 2: <think> tags in the content itself
    thinking_from_tags, answer = _parse_thinking(raw_text)

    # Use reasoning field if available, otherwise fall back to <think> tags
    thinking = reasoning_field or thinking_from_tags

    usage = {}
    if response.usage:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return ModelResponse(
        thinking=thinking,
        answer=answer,
        raw=raw_text,
        finish_reason=choice.finish_reason or "unknown",
        usage=usage,
    )


async def async_complete(
    model_id: str,
    messages: list[dict],
    *,
    temperature: float = 0.6,
    max_tokens: int = 8192,
    suppress_thinking: bool = False,
    cot_suppression_method: Optional[str] = None,
    reasoning_effort: Optional[str] = None,
    provider_order: Optional[list[str]] = None,
) -> ModelResponse:
    """Async version of complete(). Same interface, uses AsyncOpenAI.

    provider_order pins OpenRouter routing to specific providers (with
    fallbacks disabled) — providers differ on populating the `reasoning`
    field, and an unpinned route can silently return an empty CoT.
    """
    client = get_async_client()

    if suppress_thinking and cot_suppression_method:
        messages = _apply_cot_suppression(messages, cot_suppression_method)
        max_tokens = min(max_tokens, 100)

    kwargs = dict(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    extra_body: dict = {}
    if model_id in _NEEDS_REASONING_PARAM:
        extra_body["reasoning"] = {"max_tokens": max_tokens}
    elif reasoning_effort:
        extra_body["reasoning"] = {"effort": reasoning_effort}
    if provider_order:
        extra_body["provider"] = {"order": list(provider_order), "allow_fallbacks": False}
    if extra_body:
        kwargs["extra_body"] = extra_body

    response = await client.chat.completions.create(**kwargs)

    return _parse_response(response)


def _apply_cot_suppression(
    messages: list[dict], method: str
) -> list[dict]:
    """Apply CoT suppression to messages based on the method.

    Args:
        messages: Original chat messages.
        method: "no_think" (Qwen3) or "reasoning_complete" (DeepSeek R1).

    Returns:
        Modified messages with suppression applied.
    """
    messages = [dict(m) for m in messages]  # shallow copy

    if method == "no_think":
        # Qwen3: prepend /no_think to the user message
        for msg in messages:
            if msg["role"] == "user":
                msg["content"] = "/no_think " + msg["content"]
                break

    elif method == "reasoning_complete":
        # DeepSeek R1: append an assistant message to skip thinking
        messages.append({
            "role": "assistant",
            "content": "\n<think>I have finished reasoning.</think>",
        })

    return messages


def extract_answer_number(text: str) -> Optional[int]:
    """Extract the numeric answer from model output.

    Looks for "ANSWER: <number>" pattern, falling back to last integer in text.
    """
    # Try ANSWER: pattern first
    matches = re.findall(r"ANSWER\s*:\s*(\d+)", text)
    if matches:
        return int(matches[-1])

    # Fallback: last integer in text
    matches = re.findall(r"\d+", text)
    if matches:
        return int(matches[-1])

    return None
