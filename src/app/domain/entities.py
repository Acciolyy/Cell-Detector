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
class ModelViewport:
    """Maps model input space back to the region of the frame it came from.

    The model sees a downscaled (and, for ``fit-shortest``, centre-cropped)
    view of the frame. Rendering boxes correctly requires knowing both the
    scale and the crop origin, not just the model input size.
    """

    model_width: int
    model_height: int
    offset_x: int = 0
    offset_y: int = 0
    region_width: int = 0
    region_height: int = 0

    def to_frame(self, x: float, y: float) -> tuple[float, float]:
        """Convert a point in model space to frame pixel coordinates."""
        if self.model_width <= 0 or self.model_height <= 0:
            return x, y

        scale_x = self.region_width / float(self.model_width)
        scale_y = self.region_height / float(self.model_height)
        return self.offset_x + x * scale_x, self.offset_y + y * scale_y


@dataclass(slots=True)
class InferenceResult:
    """Represents normalized result returned by inference service."""

    boxes: list[BoundingBox]
    latency_ms: float | None = None
    raw: dict[str, Any] | None = None
    model_frame: Any | None = None
    viewport: ModelViewport | None = None

