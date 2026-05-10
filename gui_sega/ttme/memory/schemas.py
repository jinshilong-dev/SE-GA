from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from PIL import Image


@dataclass
class EpisodicEntry:
    observation: str
    action: str
    next_observation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EpisodicEntry":
        return cls(**d)


@dataclass
class SemanticEntry:
    description: str
    embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SemanticEntry":
        return cls(**d)


@dataclass
class ExperientialEntry:
    trajectory: str
    reflective_summary: str
    intent_embedding: List[float] = field(default_factory=list)
    task_embedding: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ExperientialEntry":
        return cls(**d)
