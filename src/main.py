"""Application entrypoint for real-time phone usage detection."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

from src.app.config.settings import (
    DEFAULT_CAPTURE_HEIGHT,
    DEFAULT_CAPTURE_WIDTH,
    AppSettings,
)
from src.app.utils.logger import get_logger, setup_logging
from src.app.utils.paths import ensure_file_exists, resolve_model_path

LOGGER = get_logger(__name__)

# Options that only mean something when the frames come from a webcam.
CAMERA_ONLY_FLAGS = (
    "--camera-index",
    "--camera-width",
    "--camera-height",
    "--camera-fourcc",
)


def _camera_flags_present(argv: Sequence[str]) -> list[str]:
    """Return the camera-only flags actually typed on the command line.

    Comparing against parser defaults would not do: ``--camera-index 0`` is
    indistinguishable from the default, so the argv is inspected directly.
    """
    return [
        flag
        for flag in CAMERA_ONLY_FLAGS
        if any(arg == flag or arg.startswith(f"{flag}=") for arg in argv)
    ]


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
        "--source",
        type=Path,
        default=None,
        metavar="PATH",
        help="Read frames from a video file instead of the webcam. Every "
        "--camera-* option is ignored in this mode.",
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
        default=DEFAULT_CAPTURE_WIDTH,
        help="Webcam width in pixels. The model input is 96x96; capturing "
        "higher than the default buys no accuracy and costs USB bandwidth.",
    )
    parser.add_argument(
        "--camera-height",
        type=int,
        default=DEFAULT_CAPTURE_HEIGHT,
        help="Webcam height in pixels.",
    )
    parser.add_argument(
        "--camera-fourcc",
        default=None,
        help="Optional 4-character capture format, e.g. MJPG. Useful when the "
        "driver defaults to uncompressed YUYV and caps the frame rate.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for detections [0.0, 1.0].",
    )
    parser.add_argument(
        "--record",
        type=Path,
        default=None,
        metavar="PATH.mp4",
        help="Write annotated frames to an mp4 file at the measured FPS.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without an OpenCV window (headless recording or profiling).",
    )
    parser.add_argument(
        "--profile-frames",
        type=int,
        default=None,
        metavar="N",
        help="Process exactly N frames, then print the stage timing report.",
    )
    parser.add_argument(
        "--fps-window",
        type=int,
        default=30,
        help="Number of frames averaged for the FPS shown on the overlay.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Log verbosity level.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    setup_logging(getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        from src.app.services.camera_service import (
            CameraReadError,
            CameraService,
            CameraUnavailableError,
        )
        from src.app.services.detection_pipeline import DetectionPipeline
        from src.app.services.eim_inference_service import (
            EimInferenceService,
            InferenceExecutionError,
            InferenceInitializationError,
        )
        from src.app.services.frame_source import (
            ThreadedFrameSource,
            VideoFileFrameSource,
            VideoSourceError,
        )
        from src.app.services.video_recorder import VideoRecordingError
    except ModuleNotFoundError as exc:
        LOGGER.error(
            "Missing runtime dependency: %s. Install requirements with: "
            "pip install -r requirements.txt",
            exc.name or exc,
        )
        raise SystemExit(1) from exc

    try:
        model_path = resolve_model_path(args.model_path)
        ensure_file_exists(model_path)

        if args.source is not None:
            ignored = _camera_flags_present(argv)
            if ignored:
                LOGGER.warning(
                    "Ignoring %s: --source reads frames from a file, not a camera.",
                    ", ".join(ignored),
                )

        settings = AppSettings(
            model_path=model_path,
            source_path=args.source,
            camera_index=args.camera_index,
            camera_width=args.camera_width,
            camera_height=args.camera_height,
            camera_fourcc=args.camera_fourcc,
            min_confidence=args.min_confidence,
            show_display=not args.no_display,
            record_path=args.record,
            fps_window=args.fps_window,
            profile_frames=args.profile_frames,
        )
        settings.validate()

        if settings.uses_video_file:
            frame_source = VideoFileFrameSource(settings.source_path)
        else:
            frame_source = ThreadedFrameSource(
                CameraService(
                    camera_index=settings.camera_index,
                    width=settings.camera_width,
                    height=settings.camera_height,
                    fourcc=settings.camera_fourcc,
                )
            )

        inference_service = EimInferenceService(
            model_path=settings.model_path,
            min_confidence=settings.min_confidence,
        )
        pipeline = DetectionPipeline(settings, frame_source, inference_service)
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
    except CameraReadError as exc:
        LOGGER.error("Webcam read failure: %s", exc)
        raise SystemExit(1) from exc
    except VideoSourceError as exc:
        LOGGER.error("Video source unusable: %s", exc)
        raise SystemExit(1) from exc
    except VideoRecordingError as exc:
        LOGGER.error("Recording failure: %s", exc)
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
