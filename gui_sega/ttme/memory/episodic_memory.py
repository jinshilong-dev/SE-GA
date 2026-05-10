from typing import List, Optional, Dict, Any
from PIL import Image

from .schemas import EpisodicEntry
from .utils import cosine_similarity


class EpisodicMemory:
    """
    Episodic Memory (M_EPI): Short-term working memory that tracks the
    immediate task progress and executed historical actions.

    Data structure: M_EPI^t = [m_k]_{k=1}^{t-1}
    where m_k = <o_k, a_k, o_{k+1}>

    Retrieval: Sliding window with fixed horizon H.
    C_epi^t = [m_k]_{k=epsilon}^{t-1}, epsilon = max(1, t-H)
    """

    def __init__(self, horizon: int = 10):
        self.horizon = horizon
        self._entries: List[EpisodicEntry] = []

    @property
    def entries(self) -> List[EpisodicEntry]:
        return self._entries

    def add(self, observation: str, action: str, next_observation: str):
        entry = EpisodicEntry(
            observation=observation,
            action=action,
            next_observation=next_observation,
        )
        self._entries.append(entry)

    def retrieve(self) -> List[EpisodicEntry]:
        t = len(self._entries)
        epsilon = max(0, t - self.horizon)
        return self._entries[epsilon:]

    def build_context(self) -> str:
        windowed = self.retrieve()
        if not windowed:
            return ""
        lines = []
        for i, entry in enumerate(windowed):
            lines.append(
                f"Step {i + 1}: Observation: {entry.observation} | "
                f"Action: {entry.action} | Result: {entry.next_observation}"
            )
        return "\n".join(lines)

    def clear(self):
        self._entries = []

    def __len__(self) -> int:
        return len(self._entries)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def from_list(self, data: List[Dict[str, Any]]):
        self._entries = [EpisodicEntry.from_dict(d) for d in data]
