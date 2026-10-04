"""
azure_llm.py — thin helper around the Azure OpenAI GPT-5.4 deployment.

Reusable for all three FaraGen-style roles:
  * task proposer   (generate support tickets)
  * user simulator  (answer the agent's clarifying questions)   [later]
  * verifier/judge  (optional LLM check)                         [later]

Config is read from azure_openai_config.json (gitignored). Supports text and
multimodal (image) messages.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

from openai import AzureOpenAI

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "azure_openai_config.json")


def _load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        cfg = json.load(f)
    if "PASTE" in cfg.get("api_key", ""):
        raise SystemExit(
            f"Set your API key in {_CONFIG_PATH} before using the Azure LLM."
        )
    return cfg


class AzureLLM:
    def __init__(self):
        self.cfg = _load_config()
        self.client = AzureOpenAI(
            api_version=self.cfg["api_version"],
            azure_endpoint=self.cfg["azure_endpoint"],
            api_key=self.cfg["api_key"],
        )
        self.deployment = self.cfg["deployment"]

    def chat(self, system: str, user: str,
             image_png: bytes | None = None, **kwargs) -> str:
        """Single-turn chat. Optionally attach a screenshot (PNG bytes)."""
        content: Any = user
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content = [
                {"type": "text", "text": user},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            **kwargs,
        )
        return resp.choices[0].message.content

    def chat_json(self, system: str, user: str, **kwargs) -> Any:
        """Chat that must return JSON (uses response_format=json_object)."""
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            **kwargs,
        )
        return json.loads(resp.choices[0].message.content)


if __name__ == "__main__":
    llm = AzureLLM()
    print(llm.chat("You are a terse assistant.", "Say 'gpt-5.4 wired up' and nothing else."))
