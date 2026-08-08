"""Webcam capture service using OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from src.app.utils.logger import get_logger

LOGGER = get_logger(__name__)


class CameraUnavailableError(RuntimeError):
    """Raised when webcam cannot be opened."""


class CameraReadError(RuntimeError):
    """Raised when a frame cannot be read from webcam."""


class CameraService:
    """Handles camera initialization and frame retrieval."""

    def __init__(
        self,
        camera_index: int = 0,
        width: int | None = None,
        height: int | None = None,
        fourcc: str | None = None,
        buffer_size: int = 1,
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._fourcc = fourcc
        self._buffer_size = buffer_size
        self._capture: cv2.VideoCapture | None = None

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def camera_index(self) -> int:
        return self._camera_index

    def open(self) -> None:
        """Open webcam capture device."""
        self._capture = cv2.VideoCapture(self._camera_index)
        if self._capture is None or not self._capture.isOpened():
            raise CameraUnavailableError(
                f"Could not open webcam index {self._camera_index}."
            )

        # Order matters: FOURCC before the frame size, because changing the
        # pixel format can reset the negotiated resolution on V4L2 backends.
        if self._fourcc is not None:
            self._capture.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc)
            )
        if self._width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))

        # A driver-side queue deeper than 1 hands us frames that were captured
        # several periods ago, adding latency without adding throughput.
        # Not every backend honours this, so failure here is not fatal.
        try:
            self._capture.set(cv2.CAP_PROP_BUFFERSIZE, float(self._buffer_size))
        except cv2.error:  # pragma: no cover - backend specific
            LOGGER.debug("Backend rejected CAP_PROP_BUFFERSIZE.", exc_info=True)

        LOGGER.info(
            "Camera %s opened at %sx%s (requested %sx%s), driver FPS %.1f.",
            self._camera_index,
            *self.actual_size(),
            self._width,
            self._height,
            self.driver_fps(),
        )

    def actual_size(self) -> tuple[int, int]:
        """Return the resolution the driver actually negotiated."""
        if self._capture is None:
            return (0, 0)
        return (
            int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def driver_fps(self) -> float:
        """Return the frame rate reported by the driver (0.0 when unknown)."""
        if self._capture is None:
            return 0.0
        try:
            return float(self._capture.get(cv2.CAP_PROP_FPS))
        except (cv2.error, TypeError, ValueError):  # pragma: no cover
            return 0.0

    def read_frame(self) -> np.ndarray:
        """Read one frame from webcam."""
        if self._capture is None:
            raise CameraReadError("Camera has not been opened.")

        ok, frame = self._capture.read()
        if not ok or frame is None:
            raise CameraReadError(
                f"Failed to read frame from webcam index {self._camera_index}."
            )
        return frame

    def release(self) -> None:
        """Release webcam resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
