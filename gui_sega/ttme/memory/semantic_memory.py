from typing import List, Optional, Dict, Any

from .schemas import SemanticEntry
from .utils import cosine_similarity, top_k_indices, EmbeddingProvider


class SemanticMemory:
    """
    Semantic Memory (M_SEM): Persistent long-term knowledge repository that
    stores abstract knowledge and universal interaction logic.

    Data structure: m_sem^i = <k_sem^i, d^i>
    - d^i: Textual description of the interaction rule
    - k_sem^i = phi(Q_hist): Vector representation (embedding)

    Retrieval: Embedding-based similarity with cosine similarity.
    S_sem(Q, m_sem^i) = phi(Q) . k_sem^i / (|phi(Q)| |k_sem^i|)

    Context construction: Top-K entries with highest relevance scores.
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        top_k: int = 3,
    ):
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self._entries: List[SemanticEntry] = []

    @property
    def entries(self) -> List[SemanticEntry]:
        return self._entries

    def add(self, description: str, embedding: Optional[List[float]] = None):
        if embedding is None:
            embedding = self.embedding_provider.get_embedding(description)
        entry = SemanticEntry(description=description, embedding=embedding)
        self._entries.append(entry)

    def add_batch(self, descriptions: List[str]):
        for desc in descriptions:
            self.add(desc)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SemanticEntry]:
        k = top_k or self.top_k
        if not self._entries:
            return []

        query_embedding = self.embedding_provider.get_embedding(query)

        scores = []
        for entry in self._entries:
            if not entry.embedding:
                scores.append(-1.0)
                continue
            try:
                sim = cosine_similarity(query_embedding, entry.embedding)
            except ValueError:
                sim = -1.0
            scores.append(sim)

        indices = top_k_indices(scores, k)
        return [self._entries[i] for i in indices]

    def build_context(self, query: str, top_k: Optional[int] = None) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return ""
        lines = ["[Semantic Memory - Universal Interaction Rules]"]
        for i, entry in enumerate(results):
            lines.append(f"Rule {i + 1}: {entry.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def from_list(self, data: List[Dict[str, Any]]):
        self._entries = [SemanticEntry.from_dict(d) for d in data]
