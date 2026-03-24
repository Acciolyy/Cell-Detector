"""Domain entities used across camera and inference layers."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class BoundingBox:
    """Represents an object detection bounding box."""

    label: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class InferenceResult:
    """Represents normalized result returned by inference service."""

    boxes: list[BoundingBox]
    latency_ms: float | None = None
    raw: dict[str, Any] | None = None
    model_frame: Any | None = None

