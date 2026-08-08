"""Edge Impulse .eim inference integration for Linux."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from src.app.domain.entities import BoundingBox, InferenceResult, ModelViewport
from src.app.utils.logger import get_logger

LOGGER = get_logger(__name__)

# Studio resize mode we can reproduce with a cheap centre crop before scaling.
RESIZE_MODE_FIT_SHORTEST = "fit-shortest"

try:
    # Official Edge Impulse Linux image runner for .eim models.
    from edge_impulse_linux.image import ImageImpulseRunner  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - environment-specific
    ImageImpulseRunner = None  # type: ignore[assignment]


class InferenceInitializationError(RuntimeError):
    """Raised when Edge Impulse runtime cannot be initialized."""


class InferenceExecutionError(RuntimeError):
    """Raised when inference call fails."""


class EimInferenceService:
    """Wrapper around Edge Impulse Linux runner for .eim models."""

    def __init__(self, model_path: Path, min_confidence: float = 0.5) -> None:
        self._model_path = model_path
        self._min_confidence = min_confidence
        self._runner: Any | None = None
        self._model_info: dict[str, Any] | None = None
        self._input_frame_size: tuple[int, int] | None = None
        self._resize_mode: str | None = None

    @property
    def model_info(self) -> dict[str, Any] | None:
        """Return metadata exposed by Edge Impulse runner after init."""
        return self._model_info

    @property
    def input_frame_size(self) -> tuple[int, int] | None:
        """Return expected model input frame size as (width, height), if known."""
        return self._input_frame_size

    @property
    def resize_mode(self) -> str | None:
        """Return the Studio resize mode the model was trained with."""
        return self._resize_mode

    def start(self) -> None:
        """Initialize runner and load .eim model."""
        if not self._model_path.exists() or not self._model_path.is_file():
            raise InferenceInitializationError(
                f"Model file not found: {self._model_path}"
            )

        if ImageImpulseRunner is None:
            raise InferenceInitializationError(
                "Edge Impulse Linux Python SDK/runtime is not available. "
                "Install a compatible Edge Impulse package for .eim execution."
            )

        try:
            self._runner = ImageImpulseRunner(str(self._model_path))
            self._model_info = self._runner.init()
        except Exception as exc:  # pragma: no cover - external runtime dependency
            raise InferenceInitializationError(
                "Could not initialize .eim runner. "
                "Verify model compatibility (Linux x86_64) and runtime dependencies."
            ) from exc

        self._input_frame_size = _extract_input_size_from_model_info(self._model_info)
        self._resize_mode = _extract_resize_mode(self._model_info)

        if self._input_frame_size is not None:
            LOGGER.info(
                "Edge Impulse model initialized (input=%sx%s, resize_mode=%s).",
                self._input_frame_size[0],
                self._input_frame_size[1],
                self._resize_mode or "unknown",
            )
        else:
            LOGGER.info(
                "Edge Impulse model initialized (input shape unavailable in metadata)."
            )

    def preprocess(self, frame_bgr: Any) -> tuple[Any, Any, ModelViewport | None]:
        """Turn a BGR frame into model features.

        Returns ``(features, model_frame, viewport)``. Split out of :meth:`infer`
        so the pipeline can time preprocessing and inference independently —
        the .eim runner only reports its own native time, which hides this cost.
        """
        if self._runner is None:
            raise InferenceExecutionError("Inference runner is not initialized.")

        try:
            # Downscale first, convert colour second. The full-resolution
            # BGR->RGB copy was pure waste: everything downstream only ever
            # looks at a 96x96 view of the frame.
            model_input, viewport = self._downscale_for_model(frame_bgr)
            frame_rgb = np.ascontiguousarray(
                cv2.cvtColor(model_input, cv2.COLOR_BGR2RGB)
            )
        except Exception as exc:  # pragma: no cover - external runtime dependency
            raise InferenceExecutionError(
                "Edge Impulse inference failed during RGB preprocessing."
            ) from exc

        try:
            # The auto/studio helper encodes features with vectorized numpy and
            # applies the same resize mode used during training. The legacy
            # get_features_from_image() loops over every pixel in Python.
            features_result = self._runner.get_features_from_image_auto_studio_settings(
                frame_rgb
            )
        except Exception as exc:  # pragma: no cover - external runtime dependency
            raise InferenceExecutionError(
                "Edge Impulse inference failed during feature extraction "
                "(runner.get_features_from_image_auto_studio_settings)."
            ) from exc

        features: Any
        model_frame: Any | None = None
        if isinstance(features_result, tuple) and len(features_result) >= 2:
            features = features_result[0]
            model_frame = features_result[1]
        else:
            features = features_result

        return features, model_frame, viewport

    def classify(
        self,
        features: Any,
        model_frame: Any | None = None,
        viewport: ModelViewport | None = None,
    ) -> InferenceResult:
        """Run detection over already-extracted features."""
        if self._runner is None:
            raise InferenceExecutionError("Inference runner is not initialized.")

        try:
            output = self._runner.classify(features)
        except Exception as exc:  # pragma: no cover - external runtime dependency
            raise InferenceExecutionError(
                "Edge Impulse inference failed during classify(features)."
            ) from exc

        if not isinstance(output, dict):
            raise InferenceExecutionError(
                "Unexpected inference output format from Edge Impulse runner."
            )

        return InferenceResult(
            boxes=_extract_bounding_boxes(
                output, min_confidence=self._min_confidence
            ),
            latency_ms=_extract_latency_ms(output),
            raw=output,
            model_frame=model_frame,
            viewport=viewport,
        )

    def infer(self, frame_bgr: Any) -> InferenceResult:
        """Run inference on a BGR frame and normalize object detection output."""
        features, model_frame, viewport = self.preprocess(frame_bgr)
        return self.classify(features, model_frame=model_frame, viewport=viewport)

    def _downscale_for_model(
        self, frame_bgr: Any
    ) -> tuple[Any, ModelViewport | None]:
        """Reduce the frame to model input size before any per-pixel work.

        For ``fit-shortest`` this reproduces the SDK's own crop-and-scale, so
        the features are unchanged, and it yields the viewport needed to map
        detections back onto the full frame. Any other resize mode is left to
        the SDK untouched rather than guessed at.
        """
        if self._input_frame_size is None or self._resize_mode != RESIZE_MODE_FIT_SHORTEST:
            return frame_bgr, None

        model_w, model_h = self._input_frame_size
        height, width = frame_bgr.shape[:2]
        if width <= 0 or height <= 0:
            return frame_bgr, None

        # fit-shortest keeps the largest centred region matching the model
        # aspect ratio, then scales it down.
        aspect = model_w / float(model_h)
        if width / float(height) > aspect:
            region_w = int(height * aspect)
            region_h = height
        else:
            region_w = width
            region_h = int(width / aspect)

        offset_x = (width - region_w) // 2
        offset_y = (height - region_h) // 2
        # Slicing produces a view, so the crop itself costs nothing.
        region = frame_bgr[offset_y : offset_y + region_h, offset_x : offset_x + region_w]
        scaled = cv2.resize(region, (model_w, model_h), interpolation=cv2.INTER_AREA)

        viewport = ModelViewport(
            model_width=model_w,
            model_height=model_h,
            offset_x=offset_x,
            offset_y=offset_y,
            region_width=region_w,
            region_height=region_h,
        )
        return scaled, viewport

    def stop(self) -> None:
        """Release runner resources."""
        if self._runner is None:
            return

        try:
            self._runner.stop()
        except Exception:  # pragma: no cover - defensive cleanup
            LOGGER.warning("Failed to stop Edge Impulse runner cleanly.", exc_info=True)
        finally:
            self._runner = None
            self._input_frame_size = None
            self._resize_mode = None


def _extract_bounding_boxes(
    output: dict[str, Any], min_confidence: float
) -> list[BoundingBox]:
    """Extract object detection boxes from Edge Impulse output payload."""
    result = output.get("result", {}) if isinstance(output, dict) else {}
    raw_boxes = result.get("bounding_boxes", [])

    boxes: list[BoundingBox] = []
    for box in raw_boxes:
        if not isinstance(box, dict):
            continue

        confidence = float(box.get("value", 0.0))
        if confidence < min_confidence:
            continue

        raw_label = str(box.get("label", "unknown"))
        boxes.append(
            BoundingBox(
                label=_normalize_phone_label(raw_label),
                confidence=confidence,
                x=int(box.get("x", 0)),
                y=int(box.get("y", 0)),
                width=int(box.get("width", 0)),
                height=int(box.get("height", 0)),
            )
        )

    return boxes


def _extract_latency_ms(output: dict[str, Any]) -> float | None:
    """Extract inference latency when available in runner output."""
    if not isinstance(output, dict):
        return None

    timing = output.get("timing")
    if isinstance(timing, dict) and "dsp" in timing and "classification" in timing:
        try:
            return float(timing["dsp"]) + float(timing["classification"])
        except (TypeError, ValueError):
            return None

    return None


def _extract_resize_mode(model_info: dict[str, Any] | None) -> str | None:
    """Read the Studio resize mode from runner metadata when reported."""
    if not isinstance(model_info, dict):
        return None

    model_parameters = model_info.get("model_parameters")
    if isinstance(model_parameters, dict):
        mode = model_parameters.get("image_resize_mode")
        if isinstance(mode, str) and mode and mode != "not-reported":
            return mode

    return None


def _extract_input_size_from_model_info(
    model_info: dict[str, Any] | None,
) -> tuple[int, int] | None:
    """Try common metadata paths to discover model input width/height."""
    if not isinstance(model_info, dict):
        return None

    candidates: list[tuple[int, int]] = []

    # Conservative probing because metadata shape can vary by SDK/runtime version.
    for key in ("input_width", "input_height"):
        if key in model_info:
            try:
                w = int(model_info.get("input_width"))
                h = int(model_info.get("input_height"))
                if w > 0 and h > 0:
                    candidates.append((w, h))
            except (TypeError, ValueError):
                pass
            break

    model_parameters = model_info.get("model_parameters")
    if isinstance(model_parameters, dict):
        try:
            w = int(model_parameters.get("image_input_width"))
            h = int(model_parameters.get("image_input_height"))
            if w > 0 and h > 0:
                candidates.append((w, h))
        except (TypeError, ValueError):
            pass

    for value in model_info.values():
        if not isinstance(value, dict):
            continue
        try:
            w = int(value.get("input_width"))
            h = int(value.get("input_height"))
            if w > 0 and h > 0:
                candidates.append((w, h))
        except (TypeError, ValueError):
            continue

    return candidates[0] if candidates else None


def _normalize_phone_label(label: str) -> str:
    """Normalize common phone-related labels for user-facing overlay."""
    normalized = label.strip().lower().replace("-", " ").replace("_", " ")
    aliases = {
        "cellphone",
        "cell phone",
        "mobile phone",
        "phone",
        "smartphone",
        "celular",
    }
    if normalized in aliases:
        return "celular"
    return label

