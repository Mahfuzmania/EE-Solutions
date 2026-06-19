from __future__ import annotations

import json
from typing import Protocol

import requests

from .config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL, REMOTE_API_KEY, REMOTE_BASE_URL, REMOTE_MODEL


class LLMClient(Protocol):
    provider: str
    model: str

    def chat(self, system: str, prompt: str) -> str:
        ...


class OllamaClient:
    provider = "ollama"

    def __init__(self, base_url: str = OLLAMA_BASE_URL, model: str = OLLAMA_MODEL):
        self.base_url = base_url.rstrip("/")
        self.model = model

    def chat(self, system: str, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
        }
        response = requests.post(self.base_url + "/api/chat", json=payload, timeout=120)
        response.raise_for_status()
        data = response.json()
        return data.get("message", {}).get("content", "")


class OpenAICompatibleClient:
    provider = "remote"

    def __init__(
        self,
        base_url: str = REMOTE_BASE_URL,
        model: str = REMOTE_MODEL,
        api_key: str = REMOTE_API_KEY,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key

    def chat(self, system: str, prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
        }
        response = requests.post(
            self.base_url + "/chat/completions",
            headers=headers,
            data=json.dumps(payload),
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]


def get_llm_client(provider: str = LLM_PROVIDER) -> LLMClient:
    provider = provider.lower()
    if provider == "ollama":
        return OllamaClient()
    if provider == "remote":
        return OpenAICompatibleClient()
    raise RuntimeError(f"Unsupported LLM provider: {provider}")
