from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml

def load_yaml_file(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def extract_json_object(text: str) -> dict[str, Any] | None:
    content = str(text or "").strip()
    if content.startswith("```"):
        content = content.strip("`").replace("json\n", "", 1).strip()
    start = content.find("{")
    end = content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def local_llm_server_available(base_url: str, timeout: float = 2.0) -> bool:
    try:
        request = urllib.request.Request(f"{base_url.rstrip('/')}/models", method="GET")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


class LocalLLMClient:
    def __init__(self, llm_config: dict[str, Any]):
        self.llm_config = dict(llm_config)
        self.base_url = str(llm_config["base_url"])
        self.model_name = str(llm_config["model_name"])
        self.api_key = str(llm_config["api_key"])
        self.temperature = float(llm_config["temperature"])
        self.max_tokens = int(llm_config["max_new_tokens"])

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if not local_llm_server_available(self.base_url):
            raise RuntimeError(f"Local LLM server is unavailable: {self.base_url}")

        from openai import OpenAI

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=120.0,
        )
        kwargs = {
            "model": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as exc:
            if tools and "tool choice requires" in str(exc):
                kwargs.pop("tools", None)
                kwargs.pop("tool_choice", None)
                response = client.chat.completions.create(**kwargs)
            else:
                raise
        message = response.choices[0].message
        tool_calls = []
        for call in list(getattr(message, "tool_calls", None) or []):
            function = getattr(call, "function", None)
            raw_arguments = str(getattr(function, "arguments", "") or "{}")
            try:
                arguments = json.loads(raw_arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": raw_arguments}
            tool_calls.append(
                {
                    "id": str(getattr(call, "id", "") or ""),
                    "name": str(getattr(function, "name", "") or ""),
                    "arguments": arguments,
                }
            )
        return {"content": str(message.content or ""), "tool_calls": tool_calls}
