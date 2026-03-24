"""Webcam capture service using OpenCV."""

from __future__ import annotations

import cv2
import numpy as np


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
    ) -> None:
        self._camera_index = camera_index
        self._width = width
        self._height = height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        """Open webcam capture device."""
        self._capture = cv2.VideoCapture(self._camera_index)
        if self._capture is None or not self._capture.isOpened():
            raise CameraUnavailableError(
                f"Could not open webcam index {self._camera_index}."
            )

        if self._width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._width))
        if self._height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._height))

        # Some camera backends ignore requested sizes; log via exception message only if read fails.

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

