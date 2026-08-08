"""Tests for the video-file frame source."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from src.app.services.frame_source import (
    FALLBACK_VIDEO_FPS,
    EndOfStream,
    VideoFileFrameSource,
    VideoSourceError,
    _resolve_frame_count,
    _resolve_video_fps,
)

FRAME_WIDTH = 64
FRAME_HEIGHT = 48
FRAME_COUNT = 25
SOURCE_FPS = 12.0


def write_video(
    path: Path,
    frames: int = FRAME_COUNT,
    fps: float = SOURCE_FPS,
) -> Path:
    """Render a synthetic clip whose frames are individually identifiable."""
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (FRAME_WIDTH, FRAME_HEIGHT),
    )
    assert writer.isOpened(), "test fixture could not open a VideoWriter"

    for index in range(frames):
        # A flat grey level per frame survives lossy compression, so the
        # decoded order can be checked without exact pixel equality.
        frame = np.full(
            (FRAME_HEIGHT, FRAME_WIDTH, 3), index * 8 + 4, dtype=np.uint8
        )
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def video_file(tmp_path: Path) -> Path:
    return write_video(tmp_path / "clip.mp4")


def drain(source: VideoFileFrameSource) -> list[np.ndarray]:
    """Read until the source reports end of stream."""
    frames: list[np.ndarray] = []
    while True:
        try:
            frames.append(source.read(timeout=5.0))
        except EndOfStream:
            return frames


class TestVideoFileFrameSource:
    def test_delivers_every_frame_without_dropping(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        try:
            frames = drain(source)
        finally:
            source.stop()

        # A dropped frame from a file is lost content, not stale content.
        assert len(frames) == FRAME_COUNT
        assert source.frames_read == FRAME_COUNT
        assert source.frames_dropped == 0

    def test_preserves_frame_order(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        try:
            levels = [int(frame.mean()) for frame in drain(source)]
        finally:
            source.stop()

        assert levels == sorted(levels)
        assert len(set(levels)) == FRAME_COUNT

    def test_end_of_file_is_not_an_error(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        try:
            drain(source)
            # Still EndOfStream on every later call, never a hang or a crash.
            with pytest.raises(EndOfStream):
                source.read(timeout=1.0)
        finally:
            source.stop()

    def test_reports_the_input_frame_rate(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        try:
            # A recording made from this file must inherit this rate, not the
            # rate at which the machine happened to process it.
            assert source.output_fps == pytest.approx(SOURCE_FPS, abs=0.5)
        finally:
            source.stop()

    def test_reports_frame_count_and_description(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        try:
            assert source.frame_count == FRAME_COUNT
            description = source.describe()
            assert "clip.mp4" in description
            assert "fps" in description
        finally:
            source.stop()

    def test_uses_the_decode_stage_name(self, video_file: Path) -> None:
        # Disk and codec cost must not be averaged into camera capture cost.
        assert VideoFileFrameSource(video_file).stage_name == "decode"

    def test_stop_releases_and_is_idempotent(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        source.stop()
        source.stop()

        assert source.is_running is False

    def test_stop_midway_does_not_hang(self, video_file: Path) -> None:
        # The decoder parks on a full queue; stopping must still drain and join.
        source = VideoFileFrameSource(video_file, queue_depth=2)
        source.start()
        source.read(timeout=5.0)
        source.stop()

        assert source.is_running is False

    def test_context_manager_releases_on_exception(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        with pytest.raises(RuntimeError):
            with source:
                raise RuntimeError("pipeline blew up")

        assert source.is_running is False

    def test_cannot_start_twice(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        source.start()
        try:
            with pytest.raises(RuntimeError):
                source.start()
        finally:
            source.stop()


class TestVideoFileErrors:
    def test_missing_file(self, tmp_path: Path) -> None:
        source = VideoFileFrameSource(tmp_path / "absent.mp4")
        with pytest.raises(VideoSourceError, match="not found"):
            source.start()

    def test_path_is_a_directory(self, tmp_path: Path) -> None:
        source = VideoFileFrameSource(tmp_path)
        with pytest.raises(VideoSourceError, match="directory"):
            source.start()

    def test_unsupported_or_corrupt_container(self, tmp_path: Path) -> None:
        broken = tmp_path / "broken.mp4"
        broken.write_bytes(b"this is definitely not a video container")

        source = VideoFileFrameSource(broken)
        with pytest.raises(VideoSourceError, match="Could not open|no readable frames"):
            source.start()

    def test_file_without_readable_frames(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        video = write_video(tmp_path / "clip.mp4")

        class EmptyCapture:
            def isOpened(self) -> bool:  # noqa: N802
                return True

            def read(self) -> tuple[bool, None]:
                return False, None

            def get(self, prop: int) -> float:
                return 0.0

            def release(self) -> None:
                self.released = True

        monkeypatch.setattr(
            "src.app.services.frame_source.cv2.VideoCapture",
            lambda *a, **k: EmptyCapture(),
        )

        source = VideoFileFrameSource(video)
        with pytest.raises(VideoSourceError, match="no readable frames"):
            source.start()

    def test_read_before_start_reports_no_decoder(self, video_file: Path) -> None:
        source = VideoFileFrameSource(video_file)
        with pytest.raises(VideoSourceError, match="no frame"):
            source.read(timeout=0.2)


class TestFrameRateResolution:
    @pytest.mark.parametrize("raw", [0.0, -1.0, float("nan"), 100000.0, None, "x"])
    def test_falls_back_when_the_container_reports_nonsense(
        self, raw: object, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("WARNING"):
            assert _resolve_video_fps(raw, Path("clip.mp4")) == FALLBACK_VIDEO_FPS

        assert "falling back" in caplog.text

    @pytest.mark.parametrize("raw", [1.0, 23.976, 30.0, 60.0])
    def test_keeps_plausible_rates(self, raw: float) -> None:
        assert _resolve_video_fps(raw, Path("clip.mp4")) == pytest.approx(raw)

    def test_frame_count_rejects_unknown_values(self) -> None:
        assert _resolve_frame_count(0.0) is None
        assert _resolve_frame_count(-1.0) is None
        assert _resolve_frame_count(120.0) == 120
