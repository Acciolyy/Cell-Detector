"""Tests for the annotated-frame recorder."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.app.services.video_recorder import VideoRecorder, VideoRecordingError

FRAME_WIDTH = 64
FRAME_HEIGHT = 48


def frame(level: int = 128) -> np.ndarray:
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), level, dtype=np.uint8)


def read_back(path: Path) -> tuple[int, float, tuple[int, int]]:
    """Return (frame count, fps, size) of a written file."""
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened(), f"could not reopen {path}"
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
            (
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            ),
        )
    finally:
        capture.release()


class TestForcedFps:
    def test_output_inherits_the_source_rate(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        recorder = VideoRecorder(output, forced_fps=12.0)
        for _ in range(10):
            # The measured processing rate is deliberately far off; it must be
            # ignored, or a file-backed demo plays back at the wrong speed.
            recorder.write(frame(), measured_fps=90.0)
        recorder.close()

        count, fps, size = read_back(output)
        assert fps == pytest.approx(12.0, abs=0.5)
        assert count == 10
        assert size == (FRAME_WIDTH, FRAME_HEIGHT)
        assert recorder.recorded_fps == pytest.approx(12.0)

    def test_forced_rate_needs_no_warmup(self, tmp_path: Path) -> None:
        recorder = VideoRecorder(tmp_path / "out.mp4", warmup_frames=30, forced_fps=25.0)
        recorder.write(frame(), measured_fps=0.0)

        # The rate is known up front, so the file opens on the first frame.
        assert recorder.is_open is True
        assert recorder.frames_written == 1
        recorder.close()


class TestMeasuredFps:
    def test_buffers_warmup_then_writes_every_frame(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        recorder = VideoRecorder(output, warmup_frames=5)
        for _ in range(12):
            recorder.write(frame(), measured_fps=20.0)
        recorder.close()

        # Warm-up frames are flushed, not discarded.
        count, fps, _ = read_back(output)
        assert count == 12
        assert fps == pytest.approx(20.0, abs=0.5)

    def test_short_run_still_produces_a_valid_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        recorder = VideoRecorder(output, warmup_frames=30)
        for _ in range(3):
            recorder.write(frame(), measured_fps=0.0)
        recorder.close(fallback_fps=15.0)

        count, fps, _ = read_back(output)
        assert count == 3
        assert fps == pytest.approx(15.0, abs=0.5)

    def test_rate_is_clamped_to_something_playable(self, tmp_path: Path) -> None:
        recorder = VideoRecorder(tmp_path / "out.mp4", warmup_frames=1)
        recorder.write(frame(), measured_fps=100000.0)
        recorder.close()

        assert recorder.recorded_fps is not None
        assert 1.0 <= recorder.recorded_fps <= 120.0

    def test_close_without_frames_writes_nothing(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        VideoRecorder(output).close()

        assert output.exists() is False


class TestRecorderErrors:
    def test_unwritable_destination_raises(self, tmp_path: Path) -> None:
        recorder = VideoRecorder(tmp_path / "missing-dir" / "out.mp4", warmup_frames=1)
        with pytest.raises(VideoRecordingError):
            recorder.write(frame(), measured_fps=20.0)

    def test_context_manager_closes_the_file(self, tmp_path: Path) -> None:
        output = tmp_path / "out.mp4"
        with VideoRecorder(output, forced_fps=10.0) as recorder:
            recorder.write(frame(), measured_fps=0.0)

        assert recorder.is_open is False
        assert read_back(output)[0] == 1
