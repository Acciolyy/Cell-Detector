"""Application entrypoint for real-time phone usage detection."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.app.config.settings import AppSettings
from src.app.utils.logger import get_logger, setup_logging
from src.app.utils.paths import ensure_file_exists, resolve_model_path

LOGGER = get_logger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real-time cellphone detection via Edge Impulse .eim model."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("model/cellphone_detector.eim"),
        help="Path to Edge Impulse .eim model file.",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV webcam index.",
    )
    parser.add_argument(
        "--camera-width",
        type=int,
        default=None,
        help="Optional webcam width in pixels.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=None,
        help="Optional webcam height in pixels.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detections [0.0, 1.0].",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log verbosity level.",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        from src.app.services.camera_service import CameraUnavailableError, CameraService
        from src.app.services.detection_pipeline import DetectionPipeline
        from src.app.services.eim_inference_service import (
            EimInferenceService,
            InferenceExecutionError,
            InferenceInitializationError,
        )
    except ModuleNotFoundError as exc:
        LOGGER.error(
            "Missing runtime dependency. Install requirements with: pip install -r requirements.txt"
        )
        raise SystemExit(1) from exc

    try:
        model_path = resolve_model_path(args.model_path)
        ensure_file_exists(model_path)

        settings = AppSettings(
            model_path=model_path,
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            min_confidence=args.min_confidence,
        )
        settings.validate()

        camera_service = CameraService(
            camera_index=settings.camera_index,
            width=settings.camera_width,
            height=settings.camera_height,
        )
        inference_service = EimInferenceService(
            model_path=settings.model_path,
            min_confidence=settings.min_confidence,
        )
        pipeline = DetectionPipeline(settings, camera_service, inference_service)
        pipeline.run()

    except FileNotFoundError as exc:
        LOGGER.error("Model file not found: %s", exc)
        raise SystemExit(1) from exc
    except ValueError as exc:
        LOGGER.error("Invalid configuration: %s", exc)
        raise SystemExit(1) from exc
    except CameraUnavailableError as exc:
        LOGGER.error("Webcam unavailable: %s", exc)
        raise SystemExit(1) from exc
    except InferenceInitializationError as exc:
        LOGGER.error("Failed to initialize Edge Impulse inference: %s", exc)
        raise SystemExit(1) from exc
    except InferenceExecutionError as exc:
        LOGGER.error("Inference execution error: %s", exc)
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        LOGGER.info("Interrupted by user.")


if __name__ == "__main__":
    main()
