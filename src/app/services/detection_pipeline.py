"""Real-time detection pipeline orchestration."""

from __future__ import annotations

import time
from typing import Any

import cv2

from src.app.config.settings import AppSettings
from src.app.domain.entities import BoundingBox, InferenceResult
from src.app.services.camera_service import CameraService
from src.app.services.eim_inference_service import EimInferenceService
from src.app.utils.logger import get_logger

LOGGER = get_logger(__name__)


class DetectionPipeline:
    """Coordinates frame capture, inference and rendering."""

    def __init__(
        self,
        settings: AppSettings,
        camera_service: CameraService,
        inference_service: EimInferenceService,
    ) -> None:
        self._settings = settings
        self._camera_service = camera_service
        self._inference_service = inference_service
        self._last_frame_time = time.perf_counter()

    def run(self) -> None:
        """Start processing webcam frames until user exits with key 'q'."""
        LOGGER.info("Starting detection pipeline. Press 'q' to exit.")

        self._camera_service.open()
        self._inference_service.start()

        try:
            while True:
                frame = self._camera_service.read_frame()
                inference = self._inference_service.infer(frame)

                fps = self._compute_fps()
                model_space_size = self._resolve_model_space_size(inference)
                if model_space_size is None:
                    model_space_size = self._inference_service.input_frame_size
                self._draw_detections(
                    frame,
                    inference.boxes,
                    model_input_size=model_space_size,
                )
                self._draw_overlay(frame, fps=fps, inference=inference)

                cv2.imshow(self._settings.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    LOGGER.info("Exit requested by user.")
                    break
        finally:
            self._inference_service.stop()
            self._camera_service.release()
            cv2.destroyAllWindows()

    def _compute_fps(self) -> float:
        """Compute smoothed FPS from current and previous frame timestamps."""
        now = time.perf_counter()
        elapsed = now - self._last_frame_time
        self._last_frame_time = now
        if elapsed <= 0:
            return 0.0
        return 1.0 / elapsed

    @staticmethod
    def _resolve_model_space_size(
        inference: InferenceResult,
    ) -> tuple[int, int] | None:
        """Resolve coordinates source size, preferring actual model frame over metadata."""
        model_frame = inference.model_frame
        if hasattr(model_frame, "shape"):
            shape = getattr(model_frame, "shape", None)
            if shape is not None and len(shape) >= 2:
                try:
                    height = int(shape[0])
                    width = int(shape[1])
                    if width > 0 and height > 0:
                        return (width, height)
                except (TypeError, ValueError):
                    pass
        return None

    @staticmethod
    def _draw_detections(
        frame: Any,
        boxes: list[BoundingBox],
        model_input_size: tuple[int, int] | None,
    ) -> None:
        """Draw all detections as bounding boxes with label and confidence."""
        frame_h, frame_w = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = DetectionPipeline._to_frame_coordinates(
                box,
                frame_width=frame_w,
                frame_height=frame_h,
                model_input_size=model_input_size,
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (20, 200, 20), 2)

            text = f"{box.label}: {box.confidence * 100:.1f}%"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (20, 200, 20),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _to_frame_coordinates(
        box: BoundingBox,
        frame_width: int,
        frame_height: int,
        model_input_size: tuple[int, int] | None,
    ) -> tuple[int, int, int, int]:
        """Convert model-space bbox to frame-space bbox, clipping to frame bounds."""
        x1 = float(box.x)
        y1 = float(box.y)
        x2 = float(box.x + box.width)
        y2 = float(box.y + box.height)

        if model_input_size is not None:
            model_w, model_h = model_input_size
            if model_w > 0 and model_h > 0:
                sx = frame_width / float(model_w)
                sy = frame_height / float(model_h)
                x1 *= sx
                x2 *= sx
                y1 *= sy
                y2 *= sy

        # Clip for safety when detections exceed frame boundaries.
        xi1 = max(0, min(int(round(x1)), frame_width - 1))
        yi1 = max(0, min(int(round(y1)), frame_height - 1))
        xi2 = max(0, min(int(round(x2)), frame_width - 1))
        yi2 = max(0, min(int(round(y2)), frame_height - 1))

        # Avoid degenerate rectangles that OpenCV may fail to render consistently.
        if xi2 <= xi1:
            xi2 = min(frame_width - 1, xi1 + 1)
        if yi2 <= yi1:
            yi2 = min(frame_height - 1, yi1 + 1)

        return xi1, yi1, xi2, yi2

    @staticmethod
    def _draw_overlay(frame: Any, fps: float, inference: InferenceResult) -> None:
        """Render global debug/monitoring overlay in frame."""
        latency_text = (
            f"Latency: {inference.latency_ms:.1f} ms"
            if inference.latency_ms is not None
            else "Latency: n/a"
        )
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            latency_text,
            (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

