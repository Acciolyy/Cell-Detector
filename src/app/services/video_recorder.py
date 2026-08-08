"""Writes annotated frames straight to an mp4 file.

Recording inside the pipeline keeps the demo material clean: no window
chrome, no cursor, no screen-recorder artefacts — just the frames the
pipeline produced, at the frame rate it actually achieved.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from src.app.utils.logger import get_logger

LOGGER = get_logger(__name__)

# mp4v is the codec that ships with stock opencv-python wheels on Linux.
DEFAULT_FOURCC = "mp4v"

# A writer opened with an absurd frame rate produces a video that plays at the
# wrong speed, so measured FPS is clamped to something a player will accept.
MIN_RECORD_FPS = 1.0
MAX_RECORD_FPS = 120.0


class VideoRecordingError(RuntimeError):
    """Raised when the output video file cannot be opened or written."""


class VideoRecorder:
    """Buffers frames until the real FPS is known, then writes them all out.

    ``cv2.VideoWriter`` needs its frame rate up front, but the honest frame
    rate is only known after the loop has been running. Frames captured during
    the warm-up are held in memory and flushed once the writer opens, so
    nothing from the start of the run is lost.
    """

    def __init__(
        self,
        output_path: Path,
        warmup_frames: int = 30,
        fourcc: str = DEFAULT_FOURCC,
        forced_fps: float | None = None,
    ) -> None:
        self._output_path = output_path
        self._warmup_frames = max(1, warmup_frames)
        self._fourcc = fourcc
        # When the source has its own frame rate (a video file), the output has
        # to inherit it. Recording a file at the machine's processing rate would
        # play the demo back fast or slow.
        self._forced_fps = forced_fps

        self._writer: cv2.VideoWriter | None = None
        self._pending: list[np.ndarray] = []
        self._frame_size: tuple[int, int] | None = None
        self._frames_written = 0
        self._recorded_fps: float | None = None

    @property
    def output_path(self) -> Path:
        return self._output_path

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def recorded_fps(self) -> float | None:
        """Frame rate the file was opened with, once known."""
        return self._recorded_fps

    @property
    def is_open(self) -> bool:
        return self._writer is not None

    def write(self, frame: np.ndarray, measured_fps: float) -> None:
        """Queue or write one annotated frame."""
        if self._writer is None:
            if self._forced_fps is not None:
                # Rate is known up front: no warm-up needed.
                self._open(frame.shape, self._forced_fps)
                self._writer.write(frame)  # type: ignore[union-attr]
                self._frames_written += 1
                return

            # Frames are reused by the pipeline downstream, so buffer copies.
            self._pending.append(frame.copy())
            if len(self._pending) >= self._warmup_frames and measured_fps > 0.0:
                self._open(frame.shape, measured_fps)
            return

        self._writer.write(frame)
        self._frames_written += 1

    def close(self, fallback_fps: float = 15.0) -> None:
        """Flush anything still buffered and release the file handle."""
        if self._writer is None and self._pending:
            # Run ended before warm-up completed; still produce a valid file.
            self._open(self._pending[0].shape, fallback_fps)

        if self._writer is not None:
            self._writer.release()
            self._writer = None
            LOGGER.info(
                "Recording saved: %s (%s frames at %.2f FPS).",
                self._output_path,
                self._frames_written,
                self._recorded_fps or 0.0,
            )

        self._pending.clear()

    def __enter__(self) -> "VideoRecorder":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def _open(self, frame_shape: tuple[int, ...], measured_fps: float) -> None:
        """Open the writer at the measured rate and flush buffered frames."""
        height, width = frame_shape[:2]
        self._frame_size = (width, height)
        self._recorded_fps = min(MAX_RECORD_FPS, max(MIN_RECORD_FPS, measured_fps))

        writer = cv2.VideoWriter(
            str(self._output_path),
            cv2.VideoWriter_fourcc(*self._fourcc),
            self._recorded_fps,
            self._frame_size,
        )
        if not writer.isOpened():
            raise VideoRecordingError(
                f"Could not open video writer for {self._output_path} "
                f"(codec {self._fourcc}, {width}x{height})."
            )

        self._writer = writer
        LOGGER.info(
            "Recording to %s at %.2f FPS (%sx%s).",
            self._output_path,
            self._recorded_fps,
            width,
            height,
        )

        for buffered in self._pending:
            self._writer.write(buffered)
            self._frames_written += 1
        self._pending.clear()
