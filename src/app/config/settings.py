"""Application settings and defaults."""

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT_DIR / "model"
DEFAULT_MODEL_PATH = MODEL_DIR / "cellphone_detector.eim"

# The model input is 96x96. Capturing at 1280x720 buys no accuracy and costs
# USB bandwidth, which is what starves the loop; 640x480 is the sane default.
DEFAULT_CAPTURE_WIDTH = 640
DEFAULT_CAPTURE_HEIGHT = 480

# Frames buffered while measuring the real FPS before the recorder is opened.
RECORDER_WARMUP_FRAMES = 30


@dataclass(slots=True)
class AppSettings:
    """Runtime settings for the real-time detection pipeline."""

    model_path: Path = DEFAULT_MODEL_PATH
    # When set, frames come from this video file and every camera option is
    # ignored. Useful where no webcam is exposed, e.g. WSL2 without V4L2.
    source_path: Path | None = None
    camera_index: int = 0
    camera_width: int | None = DEFAULT_CAPTURE_WIDTH
    camera_height: int | None = DEFAULT_CAPTURE_HEIGHT
    camera_fourcc: str | None = None
    min_confidence: float = 0.5
    window_name: str = "Edge Phone Usage Detector"
    show_display: bool = True
    record_path: Path | None = None
    fps_window: int = 30
    profile_frames: int | None = None

    @property
    def uses_video_file(self) -> bool:
        """True when frames come from a file rather than a camera."""
        return self.source_path is not None

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

        if self.camera_fourcc is not None and len(self.camera_fourcc) != 4:
            raise ValueError("camera_fourcc must be a 4-character code, e.g. MJPG")

        if self.fps_window < 1:
            raise ValueError("fps_window must be >= 1")

        if self.profile_frames is not None and self.profile_frames < 1:
            raise ValueError("profile_frames must be >= 1 when provided")

        if self.record_path is not None:
            parent = self.record_path.parent
            if parent and not parent.exists():
                raise ValueError(f"Recording directory does not exist: {parent}")

        if not self.show_display and self.profile_frames is None:
            # Headless with no frame budget only makes sense while recording;
            # otherwise the run has no visible output. A video file at least
            # ends on its own, so it does not need an artificial stop.
            if self.record_path is None and not self.uses_video_file:
                raise ValueError(
                    "--no-display requires --record or --profile-frames, "
                    "otherwise the run produces no output."
                )
