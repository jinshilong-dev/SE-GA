import math
from typing import List, Optional
from PIL import Image
import numpy as np


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    if len(vec1) != len(vec2):
        raise ValueError("Embedding vectors must be of same length")
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot_product / (norm1 * norm2)


def top_k_indices(scores: List[float], k: int) -> List[int]:
    if k <= 0:
        return []
    k = min(k, len(scores))
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in indexed[:k]]


class EmbeddingProvider:
    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "text-embedding-3-small",
    ):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def get_embedding(self, text: str) -> List[float]:
        text = text.replace("\n", " ").strip()
        if not text:
            raise ValueError("Text cannot be empty")
        response = self.client.embeddings.create(model=self.model, input=text)
        return response.data[0].embedding


class VisualEncoder:
    def __init__(
        self,
        model_name: str = "clip-ViT-B-32",
        device: str = "cuda",
    ):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model_name, device=device)
        except ImportError:
            self.model = None

    def encode_image(self, image: Image.Image) -> List[float]:
        if self.model is None:
            return []
        embedding = self.model.encode(image, convert_to_numpy=True)
        return embedding.tolist()
