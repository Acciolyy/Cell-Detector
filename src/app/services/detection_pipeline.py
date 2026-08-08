"""Real-time detection pipeline orchestration."""

from __future__ import annotations

from typing import Any

import cv2

from src.app.config.settings import RECORDER_WARMUP_FRAMES, AppSettings
from src.app.domain.entities import BoundingBox, InferenceResult, ModelViewport
from src.app.services.camera_service import CameraReadError
from src.app.services.eim_inference_service import EimInferenceService
from src.app.services.frame_source import EndOfStream, FrameSource, describe_source
from src.app.services.video_recorder import VideoRecorder
from src.app.utils.logger import get_logger
from src.app.utils.profiling import (
    STAGE_DISPLAY,
    STAGE_DRAW,
    STAGE_INFERENCE,
    STAGE_PREPROCESS,
    STAGE_RECORD,
    FrameProfiler,
    MovingAverageFps,
)

LOGGER = get_logger(__name__)

BOX_COLOR = (20, 200, 20)
HUD_COLOR = (240, 240, 240)
HUD_SHADOW = (0, 0, 0)
HUD_FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_SCALE = 0.42
HUD_LINE_HEIGHT = 15
HUD_ORIGIN = (8, 16)


class DetectionPipeline:
    """Coordinates frame capture, inference and rendering."""

    def __init__(
        self,
        settings: AppSettings,
        frame_source: FrameSource,
        inference_service: EimInferenceService,
    ) -> None:
        self._settings = settings
        self._frame_source = frame_source
        self._inference_service = inference_service
        self._fps = MovingAverageFps(window=settings.fps_window)
        self._profiler = FrameProfiler()

    @property
    def profiler(self) -> FrameProfiler:
        """Stage timings collected during the run."""
        return self._profiler

    def run(self) -> None:
        """Process webcam frames until the user exits or the budget is spent."""
        budget = self._settings.profile_frames
        LOGGER.info(
            "Starting detection pipeline (%s).",
            f"{budget} frames" if budget else "press 'q' to exit",
        )

        # Started before the frame source so a source failure still shuts the
        # runner down through the finally block below.
        self._inference_service.start()
        recorder: VideoRecorder | None = None
        try:
            self._frame_source.start()

            if self._settings.record_path is not None:
                recorder = VideoRecorder(
                    self._settings.record_path,
                    warmup_frames=RECORDER_WARMUP_FRAMES,
                    # A file-backed source dictates the output rate; a camera
                    # has none, so the measured rate is used instead.
                    forced_fps=self._frame_source.output_fps,
                )

            self._loop(recorder, budget)
        finally:
            self._shutdown(recorder)

        LOGGER.info(
            "Processed %s frames from %s (%s).",
            self._frame_source.frames_read,
            self._frame_source.describe(),
            describe_source(self._frame_source),
        )
        LOGGER.info(
            "\n%s",
            self._profiler.format_report(
                "Stage timing summary", source=self._frame_source.describe()
            ),
        )

    def _loop(self, recorder: VideoRecorder | None, budget: int | None) -> None:
        """Main frame loop, one profiled iteration per frame."""
        frames = 0
        source_stage = self._frame_source.stage_name

        while budget is None or frames < budget:
            exhausted = False
            quit_requested = False

            with self._profiler.frame_context():
                try:
                    with self._profiler.stage(source_stage):
                        frame = self._frame_source.read()
                except EndOfStream:
                    # A finite source running out is the normal way to finish.
                    exhausted = True

                if not exhausted:
                    quit_requested = self._process_frame(frame, recorder)

            if exhausted:
                # That iteration produced no frame, so it must not land in the
                # statistics: it would understate every stage's share.
                self._profiler.discard_frame(source_stage)
                LOGGER.info("Reached end of source after %s frames.", frames)
                return

            self._profiler.mark_frame_end()
            frames += 1

            if quit_requested:
                LOGGER.info("Exit requested by user.")
                return

    def _process_frame(self, frame: Any, recorder: VideoRecorder | None) -> bool:
        """Run one frame through inference, drawing and output.

        Returns True when the user asked to quit.
        """
        with self._profiler.stage(STAGE_PREPROCESS):
            features, model_frame, viewport = self._inference_service.preprocess(frame)

        with self._profiler.stage(STAGE_INFERENCE):
            inference = self._inference_service.classify(
                features, model_frame=model_frame, viewport=viewport
            )

        fps = self._fps.tick()

        with self._profiler.stage(STAGE_DRAW):
            self._draw_detections(
                frame,
                inference.boxes,
                viewport=inference.viewport,
                model_input_size=self._resolve_model_space_size(inference)
                or self._inference_service.input_frame_size,
            )
            self._draw_hud(frame, fps=fps)

        if recorder is not None:
            with self._profiler.stage(STAGE_RECORD):
                recorder.write(frame, measured_fps=fps)

        if self._settings.show_display:
            with self._profiler.stage(STAGE_DISPLAY):
                cv2.imshow(self._settings.window_name, frame)
                return cv2.waitKey(1) & 0xFF == ord("q")

        return False

    def _shutdown(self, recorder: VideoRecorder | None) -> None:
        """Release every resource, even if some of them fail to release.

        Runs from the ``finally`` of :meth:`run`, so it must survive being
        called while an exception is already propagating.
        """
        if recorder is not None:
            try:
                recorder.close(fallback_fps=max(1.0, self._fps.value()))
            except Exception:  # noqa: BLE001 - cleanup must not mask the cause
                LOGGER.warning("Failed to finalize recording.", exc_info=True)

        try:
            self._frame_source.stop()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to stop frame source cleanly.", exc_info=True)

        try:
            self._inference_service.stop()
        except Exception:  # noqa: BLE001
            LOGGER.warning("Failed to stop inference service cleanly.", exc_info=True)

        if self._settings.show_display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:  # pragma: no cover - headless backends
                LOGGER.debug("destroyAllWindows failed.", exc_info=True)

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
        viewport: ModelViewport | None = None,
        model_input_size: tuple[int, int] | None = None,
    ) -> None:
        """Draw all detections as bounding boxes with label and confidence."""
        frame_h, frame_w = frame.shape[:2]

        for box in boxes:
            x1, y1, x2, y2 = DetectionPipeline._to_frame_coordinates(
                box,
                frame_width=frame_w,
                frame_height=frame_h,
                viewport=viewport,
                model_input_size=model_input_size,
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)

            text = f"{box.label}: {box.confidence * 100:.1f}%"
            cv2.putText(
                frame,
                text,
                (x1, max(20, y1 - 8)),
                HUD_FONT,
                0.6,
                BOX_COLOR,
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _to_frame_coordinates(
        box: BoundingBox,
        frame_width: int,
        frame_height: int,
        viewport: ModelViewport | None = None,
        model_input_size: tuple[int, int] | None = None,
    ) -> tuple[int, int, int, int]:
        """Convert model-space bbox to frame-space bbox, clipping to frame bounds."""
        if viewport is not None:
            # The model only saw a centred crop; the viewport carries both the
            # scale and that crop offset, so boxes land where the object is.
            x1, y1 = viewport.to_frame(float(box.x), float(box.y))
            x2, y2 = viewport.to_frame(
                float(box.x + box.width), float(box.y + box.height)
            )
        else:
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

    def _draw_hud(self, frame: Any, fps: float) -> None:
        """Render the metrics overlay: smoothed FPS and per-stage breakdown.

        Every number is measured by this loop. The runner's own latency covers
        native inference only, which is why it is not the headline figure.
        """
        source_stage = self._frame_source.stage_name
        capture_ms = self._profiler.stats(source_stage).last_ms
        inference_ms = (
            self._profiler.stats(STAGE_PREPROCESS).last_ms
            + self._profiler.stats(STAGE_INFERENCE).last_ms
        )
        # The previous frame's draw/display cost: this frame's is still running.
        render_ms = (
            self._profiler.stats(STAGE_DRAW).last_ms
            + self._profiler.stats(STAGE_DISPLAY).last_ms
            + self._profiler.stats(STAGE_RECORD).last_ms
        )
        frame_ms = self._profiler.frame_total.last_ms

        lines = (
            f"FPS {fps:5.1f} (avg {self._fps.window})   frame {frame_ms:5.1f} ms",
            f"{source_stage[:3]} {capture_ms:5.1f}  inf {inference_ms:5.1f}  "
            f"render {render_ms:5.1f} ms",
        )
        DetectionPipeline._draw_hud_lines(frame, lines)

    @staticmethod
    def _draw_hud_lines(frame: Any, lines: tuple[str, ...]) -> None:
        """Draw small top-left text, outlined so it stays readable on any scene."""
        x, y = HUD_ORIGIN
        for index, line in enumerate(lines):
            position = (x, y + index * HUD_LINE_HEIGHT)
            cv2.putText(
                frame, line, position, HUD_FONT, HUD_SCALE, HUD_SHADOW, 3, cv2.LINE_AA
            )
            cv2.putText(
                frame, line, position, HUD_FONT, HUD_SCALE, HUD_COLOR, 1, cv2.LINE_AA
            )


__all__ = ["DetectionPipeline", "CameraReadError"]
