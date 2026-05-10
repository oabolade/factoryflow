"""Single source of truth for the CrewAI LLM client.

Defaults to OpenAI (set ``OPENAI_API_KEY``) for local dev. For the demo the
same code switches to a vLLM-served Qwen3-8B endpoint when ``OPENAI_API_BASE``
points at a localhost or AMD cloud URL — set ``CREWAI_MODEL=openai/qwen3-8b``.

All agents must call ``get_llm()`` rather than instantiating their own LLM,
so model swaps happen in one place.
"""
from __future__ import annotations

import os

import structlog
from crewai import LLM
from dotenv import load_dotenv

load_dotenv()  # pulls .env from project root if present

log = structlog.get_logger()

DEFAULT_MODEL = "gpt-4o-mini"  # cheap and fast for local dev


def get_llm() -> LLM:
    # Only route through OPENAI_API_BASE when the user has explicitly opted into
    # a non-default model (e.g. CREWAI_MODEL=openai/qwen3-8b for vLLM). Otherwise
    # a stale vLLM URL in .env would 404 every default OpenAI call.
    explicit_model = os.getenv("CREWAI_MODEL")
    model = explicit_model or DEFAULT_MODEL
    base_url = os.getenv("OPENAI_API_BASE") if explicit_model else None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set — required for CrewAI even when using vLLM"
        )

    kwargs: dict[str, object] = {"model": model, "api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    log.info(
        "llm_configured",
        component="agents.llm_config",
        model=model,
        base_url=base_url or "openai-default",
    )
    return LLM(**kwargs)
