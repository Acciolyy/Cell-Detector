"""Application settings and defaults."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "cellphone_detector.eim"


@dataclass(slots=True)
class AppSettings:
    """Runtime settings for the real-time detection pipeline."""

    model_path: Path = DEFAULT_MODEL_PATH
    camera_index: int = 0
    camera_width: int | None = None
    camera_height: int | None = None
    min_confidence: float = 0.5
    window_name: str = "Edge Phone Usage Detector"

    def validate(self) -> None:
        """Validate critical runtime inputs before app start."""
        if self.camera_index < 0:
            raise ValueError("camera_index must be >= 0")

        if self.camera_width is not None and self.camera_width <= 0:
            raise ValueError("camera_width must be > 0 when provided")

        if self.camera_height is not None and self.camera_height <= 0:
            raise ValueError("camera_height must be > 0 when provided")

        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0.0 and 1.0")

