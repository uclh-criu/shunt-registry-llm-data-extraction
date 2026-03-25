"""
LLM abstraction: settings, provider-specific clients, and factory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class LLMSettings:
    """Configuration for creating an LLM client."""

    provider: str  # "openai" | "hf" | "ollama"
    model_id: str
    # Hugging Face generation (ignored for OpenAI for now)
    hf_max_new_tokens: int = 100
    hf_temperature: float = 0.0001
    hf_do_sample: bool = True


@runtime_checkable
class LLMClient(Protocol):
    """Minimal interface for question extraction."""

    @property
    def provider(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    def generate_chat(self, messages: list[dict[str, Any]]) -> str:
        """Run chat-style generation; messages use OpenAI-style role/content dicts."""
        ...


class OpenAIClient:
    def __init__(self, settings: LLMSettings):
        if settings.provider != "openai":
            raise ValueError("OpenAIClient requires settings.provider == 'openai'")
        self._settings = settings
        # Lazy import so `hf` runs don't require the openai package
        from openai import OpenAI

        self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        print(f"Using OpenAI provider with model: {settings.model_id}")

    @property
    def provider(self) -> str:
        return self._settings.provider

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    def generate_chat(self, messages: list[dict[str, Any]]) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.model_id,
            messages=messages,
        )
        content = response.choices[0].message.content
        return (content or "").strip()


class HuggingFaceClient:
    def __init__(self, settings: LLMSettings):
        if settings.provider != "hf":
            raise ValueError("HuggingFaceClient requires settings.provider == 'hf'")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._settings = settings
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        self._tokenizer = AutoTokenizer.from_pretrained(settings.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(settings.model_id)
        self._model = self._model.to(device)
        print(f"Using HuggingFace provider with model: {settings.model_id}")

    @property
    def provider(self) -> str:
        return self._settings.provider

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    def generate_chat(self, messages: list[dict[str, Any]]) -> str:
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)

        generated_ids = self._model.generate(
            **model_inputs,
            max_new_tokens=self._settings.hf_max_new_tokens,
            temperature=self._settings.hf_temperature,
            do_sample=self._settings.hf_do_sample,
        )

        input_length = model_inputs.input_ids.shape[1]
        generated_tokens = generated_ids[0][input_length:]
        out = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return out.strip()


class OllamaClient:
    """Local (or remote) Ollama via https://github.com/ollama/ollama-python"""

    def __init__(self, settings: LLMSettings):
        if settings.provider != "ollama":
            raise ValueError("OllamaClient requires settings.provider == 'ollama'")

        from ollama import Client

        self._settings = settings
        host = os.getenv("OLLAMA_HOST", "").strip()
        self._client = Client(host=host) if host else Client()
        suffix = f" at {host}" if host else " (default host)"
        print(f"Using Ollama with model: {settings.model_id}{suffix}")

    @property
    def provider(self) -> str:
        return self._settings.provider

    @property
    def model_id(self) -> str:
        return self._settings.model_id

    def generate_chat(self, messages: list[dict[str, Any]]) -> str:
        response = self._client.chat(
            model=self._settings.model_id,
            messages=messages,
        )
        content = response.message.content
        return (content or "").strip()


def create_llm_client(settings: LLMSettings) -> LLMClient:
    """Build the appropriate client for the given settings."""
    p = settings.provider
    if p == "openai":
        return OpenAIClient(settings)
    if p == "hf":
        return HuggingFaceClient(settings)
    if p == "ollama":
        return OllamaClient(settings)
    raise ValueError(
        f"Unknown provider: {p!r}. Must be 'openai', 'hf', or 'ollama'."
    )


def llm_settings_from_config() -> LLMSettings:
    """Construct settings from src.config (single place for defaults during migration)."""
    from config import model_id, provider

    return LLMSettings(provider=provider, model_id=model_id)


def create_llm_client_from_config() -> LLMClient:
    """Convenience: create client using provider/model_id from config."""
    return create_llm_client(llm_settings_from_config())
