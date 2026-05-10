import json
import os
from typing import List, Optional, Dict, Any
from PIL import Image

from .episodic_memory import EpisodicMemory
from .semantic_memory import SemanticMemory
from .experiential_memory import ExperientialMemory
from .utils import EmbeddingProvider, VisualEncoder


class TTMEMemory:
    """
    Test-Time Memory Extension (TTME)

    Hierarchical memory repository M = (M_EPI, M_SEM, M_EXP):
    - M_EPI: Episodic memory (short-term, sliding window)
    - M_SEM: Semantic memory (long-term, embedding-based retrieval)
    - M_EXP: Experiential memory (historical trajectories, hybrid retrieval)

    At each decision step, TTME integrates the three memory contexts:
    - C_epi: Recent action trajectory
    - C_sem: Universal interaction rules
    - C_exp: Reflective summaries of past strategies

    These are collectively incorporated as M_retrieved alongside the current
    visual observation o_t and user instruction Q.
    """

    def __init__(
        self,
        episodic_horizon: int = 10,
        semantic_top_k: int = 3,
        experiential_top_k: int = 3,
        experiential_lambda: float = 0.7,
        embedding_api_key: str = None,
        embedding_base_url: str = None,
        embedding_model: str = "text-embedding-3-small",
        visual_encoder_name: str = "clip-ViT-B-32",
        visual_device: str = "cuda",
        storage_path: Optional[str] = None,
    ):
        self.embedding_provider = EmbeddingProvider(
            api_key=embedding_api_key,
            base_url=embedding_base_url,
            model=embedding_model,
        )

        visual_encoder = None
        try:
            visual_encoder = VisualEncoder(
                model_name=visual_encoder_name,
                device=visual_device,
            )
        except Exception:
            pass

        self.episodic = EpisodicMemory(horizon=episodic_horizon)
        self.semantic = SemanticMemory(
            embedding_provider=self.embedding_provider,
            top_k=semantic_top_k,
        )
        self.experiential = ExperientialMemory(
            embedding_provider=self.embedding_provider,
            visual_encoder=visual_encoder,
            lambda_weight=experiential_lambda,
            top_k=experiential_top_k,
        )

        self.storage_path = storage_path

    def record_step(self, observation: str, action: str, next_observation: str):
        self.episodic.add(observation, action, next_observation)

    def add_semantic_rule(self, description: str, embedding: Optional[List[float]] = None):
        self.semantic.add(description, embedding)

    def add_experience(
        self,
        trajectory: str,
        reflective_summary: str,
        intent_embedding: Optional[List[float]] = None,
        task_embedding: Optional[List[float]] = None,
    ):
        self.experiential.add(trajectory, reflective_summary, intent_embedding, task_embedding)

    def retrieve_context(
        self,
        query: str,
        observation: Optional[Image.Image] = None,
    ) -> Dict[str, str]:
        c_epi = self.episodic.build_context()
        c_sem = self.semantic.build_context(query)
        c_exp = self.experiential.build_context(query, observation)

        return {
            "episodic_context": c_epi,
            "semantic_context": c_sem,
            "experiential_context": c_exp,
        }

    def build_retrieved_memory(
        self,
        query: str,
        observation: Optional[Image.Image] = None,
    ) -> str:
        contexts = self.retrieve_context(query, observation)
        parts = []

        if contexts["episodic_context"]:
            parts.append(contexts["episodic_context"])
        if contexts["semantic_context"]:
            parts.append(contexts["semantic_context"])
        if contexts["experiential_context"]:
            parts.append(contexts["experiential_context"])

        if not parts:
            return ""

        header = "[Retrieved Memory M_retrieved]"
        return header + "\n\n" + "\n\n".join(parts)

    def save(self, path: Optional[str] = None):
        save_path = path or self.storage_path
        if save_path is None:
            raise ValueError("No storage path specified")

        data = {
            "episodic": self.episodic.to_list(),
            "semantic": self.semantic.to_list(),
            "experiential": self.experiential.to_list(),
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, path: Optional[str] = None):
        load_path = path or self.storage_path
        if load_path is None:
            raise ValueError("No storage path specified")
        if not os.path.exists(load_path):
            return

        with open(load_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if "episodic" in data:
            self.episodic.from_list(data["episodic"])
        if "semantic" in data:
            self.semantic.from_list(data["semantic"])
        if "experiential" in data:
            self.experiential.from_list(data["experiential"])

    def clear_episodic(self):
        self.episodic.clear()

    def __repr__(self) -> str:
        return (
            f"TTMEMemory("
            f"episodic={len(self.episodic)}, "
            f"semantic={len(self.semantic)}, "
            f"experiential={len(self.experiential)})"
        )
