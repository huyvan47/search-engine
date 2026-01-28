# rag/conversation_state.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class Turn:
    role: str          # "user" | "assistant"
    content: str
    ts: float

class ConversationStateStore:
    """
    Lưu hội thoại NGẮN HẠN theo user_id.
    In-memory (dev). Production: thay bằng Redis.
    """
    def __init__(self, max_turns: int = 8, ttl_seconds: int = 60 * 60 * 6):
        self.max_turns = max_turns
        self.ttl = ttl_seconds
        self._store: Dict[str, List[Turn]] = {}
        self._last_seen: Dict[str, float] = {}

    def _gc_if_needed(self, user_id: str):
        now = time.time()
        last = self._last_seen.get(user_id)
        if last is None:
            return
        if now - last > self.ttl:
            self._store.pop(user_id, None)
            self._last_seen.pop(user_id, None)

    def get_turns(self, user_id: str) -> List[Turn]:
        self._gc_if_needed(user_id)
        return list(self._store.get(user_id, []))

    def append(self, user_id: str, role: str, content: str):
        self._gc_if_needed(user_id)
        now = time.time()
        turns = self._store.get(user_id, [])
        turns.append(Turn(role=role, content=content, ts=now))
        turns = turns[-self.max_turns :]
        self._store[user_id] = turns
        self._last_seen[user_id] = now

conversation_state = ConversationStateStore(max_turns=8, ttl_seconds=6*60*60)
