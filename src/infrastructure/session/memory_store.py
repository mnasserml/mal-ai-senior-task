from typing import List, Dict
from src.domain.ports import ISessionStore
from src.domain.models import ChatMessage

class InMemorySessionStore(ISessionStore):
    def __init__(self, max_history_turns: int = 5):
        self.max_history_turns = max_history_turns
        self._store: Dict[str, List[ChatMessage]] = {}

    def get_history(self, session_id: str, limit: int = 5) -> List[ChatMessage]:
        history = self._store.get(session_id, [])
        effective_limit = min(limit, self.max_history_turns)
        return history[-effective_limit:]

    def add_message(self, session_id: str, message: ChatMessage) -> None:
        if session_id not in self._store:
            self._store[session_id] = []
        self._store[session_id].append(message)
