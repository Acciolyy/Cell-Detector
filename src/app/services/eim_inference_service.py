"""Edge Impulse .eim inference integration for Linux."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from src.app.domain.entities import BoundingBox, InferenceResult
from src.app.utils.logger import get_logger

LOGGER = get_logger(__name__)

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

    @property
    def model_info(self) -> dict[str, Any] | None:
        """Return metadata exposed by Edge Impulse runner after init."""
        return self._model_info

    @property
    def input_frame_size(self) -> tuple[int, int] | None:
        """Return expected model input frame size as (width, height), if known."""
        return self._input_frame_size

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

        if self._input_frame_size is not None:
            LOGGER.info(
                "Edge Impulse model initialized (input=%sx%s).",
                self._input_frame_size[0],
                self._input_frame_size[1],
            )
        else:
            LOGGER.info(
                "Edge Impulse model initialized (input shape unavailable in metadata)."
            )

    def infer(self, frame_bgr: Any) -> InferenceResult:
        """Run inference on a BGR frame and normalize object detection output."""
        if self._runner is None:
            raise InferenceExecutionError("Inference runner is not initialized.")

        # Stage 1: OpenCV camera frames are BGR; Edge Impulse expects RGB image input.
        try:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = np.ascontiguousarray(frame_rgb)
        except Exception as exc:  # pragma: no cover - external runtime dependency
            raise InferenceExecutionError(
                "Edge Impulse inference failed during RGB preprocessing."
            ) from exc

        # Stage 2: Convert image into model features using the SDK helper.
        try:
            features_result = self._runner.get_features_from_image(frame_rgb)
        except Exception as exc:  # pragma: no cover - external runtime dependency
            raise InferenceExecutionError(
                "Edge Impulse inference failed during feature extraction "
                "(runner.get_features_from_image)."
            ) from exc

        features: Any
        model_frame: Any | None = None
        if isinstance(features_result, tuple) and len(features_result) >= 2:
            features = features_result[0]
            model_frame = features_result[1]
        else:
            features = features_result

        # Stage 3: Run classification/detection over extracted features.
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

        latency_ms = _extract_latency_ms(output)
        boxes = _extract_bounding_boxes(output, min_confidence=self._min_confidence)
        return InferenceResult(
            boxes=boxes,
            latency_ms=latency_ms,
            raw=output,
            model_frame=model_frame,
        )

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

