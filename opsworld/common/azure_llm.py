"""
azure_llm.py — Azure GPT-5.4 helper for OpsWorld (solver / proposer / simulator).

Reads config from the first of:
  $AZURE_OPENAI_CONFIG
  opsworld/azure_openai_config.json
  support_env/azure_openai_config.json      (reuses the key you already set)
Supports text + multimodal (screenshot) chat.
"""
from __future__ import annotations

import base64
import json
import os
from typing import Any

from openai import AzureOpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [
    os.environ.get("AZURE_OPENAI_CONFIG", ""),
    os.path.join(_HERE, "..", "azure_openai_config.json"),
    os.path.join(_HERE, "..", "..", "support_env", "azure_openai_config.json"),
]


def _load_config() -> dict:
    for p in _CANDIDATES:
        if p and os.path.exists(p):
            cfg = json.load(open(p))
            if "PASTE" in cfg.get("api_key", ""):
                continue
            return cfg
    raise SystemExit("No usable azure_openai_config.json found (checked: "
                     + ", ".join(c for c in _CANDIDATES if c) + ")")


class AzureLLM:
    def __init__(self):
        self.cfg = _load_config()
        self.client = AzureOpenAI(
            api_version=self.cfg["api_version"],
            azure_endpoint=self.cfg["azure_endpoint"],
            api_key=self.cfg["api_key"],
        )
        self.deployment = self.cfg["deployment"]

    def chat_json(self, system: str, user_text: str,
                  image_png: bytes | None = None, **kwargs) -> Any:
        """Chat returning parsed JSON. Optionally include a screenshot."""
        content: Any = user_text
        if image_png is not None:
            b64 = base64.b64encode(image_png).decode()
            content = [
                {"type": "text", "text": user_text},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]
        resp = self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": content}],
            response_format={"type": "json_object"},
            **kwargs,
        )
        return json.loads(resp.choices[0].message.content)


if __name__ == "__main__":
    llm = AzureLLM()
    print(llm.chat_json("Return JSON.", "Say {\"ok\": true} and nothing else."))
