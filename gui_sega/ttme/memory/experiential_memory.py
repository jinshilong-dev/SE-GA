from typing import List, Optional, Dict, Any
from PIL import Image

from .schemas import ExperientialEntry
from .utils import cosine_similarity, top_k_indices, EmbeddingProvider, VisualEncoder


class ExperientialMemory:
    """
    Experiential Memory (M_EXP): Reference repository that stores historical
    trajectories from similar previously executed tasks.

    Data structure: m_exp^i = <tau^i, g(tau^i), k_intent^i, k_task^i>
    - tau^i: The recorded raw trajectory
    - g(tau^i): A reflective summary synthesized by the agent
    - k_intent^i: Embedding for intent (text-based)
    - k_task^i: Embedding for task (visual-based)

    Retrieval: Hybrid retrieval mechanism.
    S_exp(Q, o_t) = lambda * Sim(phi(Q), k_intent^i) + (1 - lambda) * Sim(psi(o_t), k_task^i)

    Context construction: Top-K entries with highest scores, extracting g(tau^i).
    """

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        visual_encoder: Optional[VisualEncoder] = None,
        lambda_weight: float = 0.7,
        top_k: int = 3,
    ):
        self.embedding_provider = embedding_provider
        self.visual_encoder = visual_encoder
        self.lambda_weight = lambda_weight
        self.top_k = top_k
        self._entries: List[ExperientialEntry] = []

    @property
    def entries(self) -> List[ExperientialEntry]:
        return self._entries

    def add(
        self,
        trajectory: str,
        reflective_summary: str,
        intent_embedding: Optional[List[float]] = None,
        task_embedding: Optional[List[float]] = None,
    ):
        if intent_embedding is None:
            intent_embedding = self.embedding_provider.get_embedding(trajectory)
        entry = ExperientialEntry(
            trajectory=trajectory,
            reflective_summary=reflective_summary,
            intent_embedding=intent_embedding,
            task_embedding=task_embedding or [],
        )
        self._entries.append(entry)

    def retrieve(
        self,
        query: str,
        observation: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
    ) -> List[ExperientialEntry]:
        k = top_k or self.top_k
        if not self._entries:
            return []

        query_embedding = self.embedding_provider.get_embedding(query)

        obs_embedding = None
        if observation is not None and self.visual_encoder is not None:
            obs_embedding = self.visual_encoder.encode_image(observation)

        scores = []
        for entry in self._entries:
            intent_score = 0.0
            task_score = 0.0

            if entry.intent_embedding:
                try:
                    intent_score = cosine_similarity(query_embedding, entry.intent_embedding)
                except ValueError:
                    intent_score = 0.0

            if obs_embedding is not None and entry.task_embedding:
                try:
                    task_score = cosine_similarity(obs_embedding, entry.task_embedding)
                except ValueError:
                    task_score = 0.0

            if obs_embedding is not None and entry.task_embedding:
                final_score = self.lambda_weight * intent_score + (1 - self.lambda_weight) * task_score
            else:
                final_score = intent_score

            scores.append(final_score)

        indices = top_k_indices(scores, k)
        return [self._entries[i] for i in indices]

    def build_context(
        self,
        query: str,
        observation: Optional[Image.Image] = None,
        top_k: Optional[int] = None,
    ) -> str:
        results = self.retrieve(query, observation, top_k)
        if not results:
            return ""
        lines = ["[Experiential Memory - Past Task Strategies]"]
        for i, entry in enumerate(results):
            lines.append(f"Experience {i + 1}: {entry.reflective_summary}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)

    def to_list(self) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._entries]

    def from_list(self, data: List[Dict[str, Any]]):
        self._entries = [ExperientialEntry.from_dict(d) for d in data]
