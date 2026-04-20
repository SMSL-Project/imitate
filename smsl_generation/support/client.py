from __future__ import annotations

import os
from abc import ABC, abstractmethod

from smsl_generation.support.env import load_package_env


# ---------------------------------------------------------------------------
# Provider auto-detection from model name
# ---------------------------------------------------------------------------

_OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")
_GEMINI_PREFIXES = ("gemini-",)
_CLAUDE_PREFIXES = ("claude-",)

SUPPORTED_PROVIDERS = {"auto", "openai", "gemini", "claude"}


def _detect_provider(model: str) -> str:
    """Infer the provider from a model name string."""
    lower = model.lower()
    if any(lower.startswith(p) for p in _OPENAI_PREFIXES):
        return "openai"
    if any(lower.startswith(p) for p in _GEMINI_PREFIXES):
        return "gemini"
    if any(lower.startswith(p) for p in _CLAUDE_PREFIXES):
        return "claude"
    raise ValueError(
        "Cannot auto-detect provider for model `{}`. "
        "Set `provider` explicitly in your config to one of: openai, gemini, claude.".format(model)
    )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ChatClient(ABC):
    """Provider-agnostic chat-completion interface."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

# Models that require the Responses API instead of Chat Completions
_RESPONSES_API_MODELS = {"gpt-5.3-codex"}

# Models that use max_completion_tokens instead of max_tokens
_MAX_COMPLETION_TOKENS_MODELS = {"gpt-5.4", "gpt-5.3-codex"}


class OpenAIChatClient(ChatClient):
    def __init__(self, model: str, temperature: float, max_tokens: int = 16000) -> None:
        load_package_env()

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError("Install `openai` before using an OpenAI model.") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set `OPENAI_API_KEY` in `smsl_generation/.env` or in your shell environment.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        if self.model in _RESPONSES_API_MODELS:
            return self._complete_responses(system_prompt, user_prompt)
        return self._complete_chat(system_prompt, user_prompt)

    def _complete_chat(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        if self.model in _MAX_COMPLETION_TOKENS_MODELS:
            kwargs["max_completion_tokens"] = self.max_tokens
        else:
            kwargs["max_tokens"] = self.max_tokens

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def _complete_responses(self, system_prompt: str, user_prompt: str) -> str:
        kwargs = dict(
            model=self.model,
            temperature=self.temperature,
            instructions=system_prompt,
            input=user_prompt,
        )
        if self.model in _MAX_COMPLETION_TOKENS_MODELS:
            kwargs["max_output_tokens"] = self.max_tokens

        response = self.client.responses.create(**kwargs)
        return response.output_text or ""


# ---------------------------------------------------------------------------
# Google Gemini
# ---------------------------------------------------------------------------

class GeminiChatClient(ChatClient):
    def __init__(self, model: str, temperature: float, max_tokens: int = 16000) -> None:
        load_package_env()

        try:
            from google import genai  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install `google-genai` before using a Gemini model: pip install google-genai"
            ) from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set `GEMINI_API_KEY` in `smsl_generation/.env` or in your shell environment.")

        from google.genai import Client
        self.client = Client(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from google.genai.types import GenerateContentConfig

        config = GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return response.text or ""


# ---------------------------------------------------------------------------
# Anthropic Claude
# ---------------------------------------------------------------------------

class ClaudeChatClient(ChatClient):
    def __init__(self, model: str, temperature: float, max_tokens: int = 16000) -> None:
        load_package_env()

        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Install `anthropic` before using a Claude model: pip install anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Set `ANTHROPIC_API_KEY` in `smsl_generation/.env` or in your shell environment.")

        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return response.content[0].text or ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_CLIENT_CLASSES = {
    "openai": OpenAIChatClient,
    "gemini": GeminiChatClient,
    "claude": ClaudeChatClient,
}


def create_chat_client(
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int = 16000,
) -> ChatClient:
    """Create a ChatClient for the given provider (or auto-detect from model name)."""
    resolved = provider if provider != "auto" else _detect_provider(model)
    if resolved not in _CLIENT_CLASSES:
        raise ValueError(
            "Unsupported provider `{}`. Choose from: {}".format(
                resolved, ", ".join(sorted(_CLIENT_CLASSES))
            )
        )
    return _CLIENT_CLASSES[resolved](model=model, temperature=temperature, max_tokens=max_tokens)
