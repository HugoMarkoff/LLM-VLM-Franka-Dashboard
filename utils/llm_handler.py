"""utils/llm_handler.py
A thin helper around the remote /chat and /chat‑stream endpoints that:
  • Streams the *thinking* tokens so the dashboard can show them live
    (Window 7 “LLM Reasoning”).
  • Collects the final answer so the normal chat pane (Window 4) gets one
    clean reply when generation ends.
  • Runs a crude NLP pass to decide whether the user’s message describes an
    **ACTION** (pick‑and‑place) or ordinary chit‑chat, and—if an action—tries
    to extract ``{object}`` and (optional) ``{placement}`` phrases that we can
    forward to RoboPoint.

Typical usage from *app.py*::

    from utils.llm_handler import LLMHandler

    llm = LLMHandler(
        stream_url="https://<ngrok>/chat-stream",   # SSE endpoint
        full_url  ="https://<ngrok>/chat"           # full‑text fallback
    )

    thinking, answer, intent = llm.process(text)
    # → thinking  = list[str] (token fragments)
    #   answer    = str       (full detokenised reply)
    #   intent    = {
    #        "type"     : "action"|"chat",
    #        "object"   : str|None,
    #        "placement": str|None
    #     }
"""
from __future__ import annotations

import asyncio
import httpx
import re
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional

load_dotenv()
NGROK_BASE_URL = os.getenv("NGROK_BASE_URL", "").rstrip('/')

# -----------------------------------------------------------------------------
#  Intent / slot‑filling helpers
# -----------------------------------------------------------------------------
ACTION_VERBS = [
    "pick", "pickup", "grab", "take", "lift",
    "place", "put", "drop", "release",
]

PLACEMENT_PREPS = [
    "in", "into", "on", "onto", "at", "to", "inside", "within"
]

OBJ_REGEX = re.compile(r"pick(?:\s+up)?\s+the\s+([a-zA-Z0-9 _-]+?)\b", re.I)
PLACE_REGEX = re.compile(r"(?:" + "|".join(PLACEMENT_PREPS) + r")\s+the\s+([a-zA-Z0-9 _-]+?)\b", re.I)


def _extract_action_slots(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (object, placement) or (None, None) if not found."""
    obj = None
    placement = None

    # Object—look after the verb
    m = OBJ_REGEX.search(text)
    if m:
        obj = m.group(1).strip()

    # Placement—look after a preposition
    p = PLACE_REGEX.search(text)
    if p:
        placement = p.group(1).strip()

    return obj, placement


def _looks_like_action(text: str) -> bool:
    t = text.lower()
    return any(v in t for v in ACTION_VERBS)


# -----------------------------------------------------------------------------
#  LLM Handler class
# -----------------------------------------------------------------------------
class LLMHandler:
    """Handler for Qwen3 Docker endpoints"""

    def __init__(self, stream_url=None, full_url=None):
        self.stream_url = stream_url or f"{NGROK_BASE_URL}/qwen3/chat-stream"
        self.full_url = full_url or f"{NGROK_BASE_URL}/qwen3/chat"

    def process(self, prompt: str) -> Tuple[List[str], str, Dict[str, Optional[str]]]:
        """Process with new Qwen3 format"""
        thinking, answer = asyncio.run(self._stream_and_collect(prompt))
        
        # Simple intent detection
        if any(verb in prompt.lower() for verb in ["pick", "place", "grab", "put"]):
            intent = {"type": "action", "object": "detected", "placement": None}
        else:
            intent = {"type": "chat", "object": None, "placement": None}
            
        return thinking, answer, intent

    async def _stream_and_collect(self, prompt: str) -> Tuple[List[str], str]:
        """Stream from Qwen3 endpoint"""
        thinking = []
        answer_parts = []

        payload = {
            "instructions": "default",
            "message": prompt,
            "session_id": "default"
        }

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                async with client.stream("POST", self.stream_url, json=payload) as r:
                    if not r.is_success:
                        raise Exception(f"Stream request failed: {r.status_code}")
                        
                    async for raw in r.aiter_lines():
                        if not raw or not raw.startswith("data:"):
                            continue
                        token = raw[5:].lstrip()
                        if token == "[DONE]":
                            break
                        thinking.append(token)
                        answer_parts.append(token)
                        
        except Exception as e:
            print(f"[LLM] Stream error: {e}, trying fallback")
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(self.full_url, json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    answer = data.get("result", "")
                    return [answer], answer
            except Exception as fallback_e:
                error_msg = f"LLM error: {fallback_e}"
                return [error_msg], error_msg

        answer = "".join(answer_parts)
        return thinking, answer