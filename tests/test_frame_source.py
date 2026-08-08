"""Tests for the threaded capture producer, with cv2.VideoCapture mocked."""

from __future__ import annotations

import threading

import cv2
import numpy as np
import pytest

from src.app.services import camera_service as camera_module
from src.app.services.camera_service import (
    CameraReadError,
    CameraService,
    CameraUnavailableError,
)
from src.app.services.frame_source import ThreadedFrameSource

WAIT_TIMEOUT_S = 5.0


class FakeVideoCapture:
    """Stand-in for cv2.VideoCapture with a controllable frame supply.

    Produces ``max_frames`` distinct frames (the frame index is stored in the
    first pixel) and then blocks on ``gate`` so tests can assert on an exact,
    race-free number of captured frames.
    """

    def __init__(self, index: int = 0, max_frames: int = 5, opened: bool = True) -> None:
        self.index = index
        self.max_frames = max_frames
        self.props: dict[int, float] = {}
        self.released = False
        self.gate = threading.Event()
        self.reached_limit = threading.Event()
        self.fail_reads = False
        self._produced = 0
        self._opened = opened

    def isOpened(self) -> bool:  # noqa: N802 - mirrors the OpenCV API
        return self._opened

    def set(self, prop: int, value: float) -> bool:  # noqa: A003
        self.props[prop] = value
        return True

    def get(self, prop: int) -> float:
        return self.props.get(prop, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.fail_reads:
            return False, None

        if self._produced >= self.max_frames:
            self.reached_limit.set()
            self.gate.wait(timeout=WAIT_TIMEOUT_S)
            return False, None

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[0, 0, 0] = self._produced
        self._produced += 1
        return True, frame

    def release(self) -> None:
        self.released = True
        self.gate.set()


@pytest.fixture
def fake_capture(monkeypatch: pytest.MonkeyPatch) -> FakeVideoCapture:
    """Install a FakeVideoCapture in place of cv2.VideoCapture."""
    capture = FakeVideoCapture()

    def factory(index: int = 0, *args: object, **kwargs: object) -> FakeVideoCapture:
        capture.index = index
        return capture

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)
    yield capture
    # Never leave a producer thread parked inside read().
    capture.gate.set()


def frame_index(frame: np.ndarray) -> int:
    return int(frame[0, 0, 0])


class TestCameraService:
    def test_open_requests_single_frame_driver_buffer(
        self, fake_capture: FakeVideoCapture
    ) -> None:
        service = CameraService(camera_index=0, width=640, height=480)
        service.open()

        # A deeper driver queue only adds latency: it hands back frames that
        # were captured several periods ago.
        assert fake_capture.props[cv2.CAP_PROP_BUFFERSIZE] == 1.0
        assert fake_capture.props[cv2.CAP_PROP_FRAME_WIDTH] == 640.0
        assert fake_capture.props[cv2.CAP_PROP_FRAME_HEIGHT] == 480.0

    def test_open_raises_when_device_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        closed = FakeVideoCapture(opened=False)
        monkeypatch.setattr(
            camera_module.cv2, "VideoCapture", lambda *a, **k: closed
        )

        with pytest.raises(CameraUnavailableError):
            CameraService(camera_index=3).open()

    def test_release_is_idempotent(self, fake_capture: FakeVideoCapture) -> None:
        service = CameraService()
        service.open()
        service.release()
        service.release()

        assert fake_capture.released is True
        assert service.is_open is False

    def test_read_frame_raises_before_open(self) -> None:
        with pytest.raises(CameraReadError):
            CameraService().read_frame()


class TestThreadedFrameSource:
    def test_keeps_only_the_newest_frame(
        self, fake_capture: FakeVideoCapture
    ) -> None:
        source = ThreadedFrameSource(CameraService())
        source.start()
        try:
            # Let the producer run ahead while the consumer does nothing.
            assert fake_capture.reached_limit.wait(WAIT_TIMEOUT_S)

            frame = source.read(timeout=WAIT_TIMEOUT_S)

            # Five frames captured, only the last survives; four were dropped
            # rather than queued, so the consumer never sees stale video.
            assert source.frames_captured == 5
            assert source.frames_dropped == 4
            assert frame_index(frame) == 4
        finally:
            source.stop()

    def test_slot_is_emptied_after_a_read(
        self, fake_capture: FakeVideoCapture
    ) -> None:
        source = ThreadedFrameSource(CameraService())
        source.start()
        try:
            assert fake_capture.reached_limit.wait(WAIT_TIMEOUT_S)
            source.read(timeout=WAIT_TIMEOUT_S)

            # Producer is parked, so nothing can refill the slot: the second
            # read must time out instead of handing back the same frame twice.
            with pytest.raises(CameraReadError):
                source.read(timeout=0.2)
        finally:
            source.stop()

    def test_read_waits_for_the_first_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture = FakeVideoCapture(max_frames=1)
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a, **k: capture)

        source = ThreadedFrameSource(CameraService())
        source.start()
        try:
            assert frame_index(source.read(timeout=WAIT_TIMEOUT_S)) == 0
        finally:
            capture.gate.set()
            source.stop()

    def test_stop_joins_the_thread_and_releases_the_camera(
        self, fake_capture: FakeVideoCapture
    ) -> None:
        source = ThreadedFrameSource(CameraService())
        source.start()
        assert source.is_running is True

        assert fake_capture.reached_limit.wait(WAIT_TIMEOUT_S)
        fake_capture.gate.set()  # unblock the producer so it can observe stop
        source.stop()

        assert source.is_running is False
        assert fake_capture.released is True

    def test_stop_is_idempotent(self, fake_capture: FakeVideoCapture) -> None:
        source = ThreadedFrameSource(CameraService())
        source.start()
        fake_capture.gate.set()
        source.stop()
        source.stop()

        assert source.is_running is False

    def test_context_manager_stops_on_exception(
        self, fake_capture: FakeVideoCapture
    ) -> None:
        with pytest.raises(RuntimeError):
            with ThreadedFrameSource(CameraService()) as source:
                assert source.is_running is True
                fake_capture.gate.set()
                raise RuntimeError("pipeline blew up")

        assert fake_capture.released is True

    def test_cannot_start_twice(self, fake_capture: FakeVideoCapture) -> None:
        source = ThreadedFrameSource(CameraService())
        source.start()
        try:
            with pytest.raises(RuntimeError):
                source.start()
        finally:
            fake_capture.gate.set()
            source.stop()

    def test_persistent_read_failure_surfaces_to_the_consumer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture = FakeVideoCapture()
        capture.fail_reads = True
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a, **k: capture)

        source = ThreadedFrameSource(CameraService(), max_read_failures=3)
        source.start()
        try:
            with pytest.raises(CameraReadError):
                source.read(timeout=WAIT_TIMEOUT_S)
        finally:
            source.stop()

    def test_transient_read_failures_do_not_kill_the_producer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        capture = FakeVideoCapture(max_frames=2)
        original_read = capture.read
        state = {"calls": 0}

        def flaky_read() -> tuple[bool, np.ndarray | None]:
            state["calls"] += 1
            if state["calls"] == 1:
                return False, None  # single glitch, e.g. USB re-enumeration
            return original_read()

        monkeypatch.setattr(capture, "read", flaky_read)
        monkeypatch.setattr(camera_module.cv2, "VideoCapture", lambda *a, **k: capture)

        source = ThreadedFrameSource(CameraService(), max_read_failures=10)
        source.start()
        try:
            assert source.read(timeout=WAIT_TIMEOUT_S) is not None
        finally:
            capture.gate.set()
            source.stop()

    def test_read_after_stop_raises(self, fake_capture: FakeVideoCapture) -> None:
        source = ThreadedFrameSource(CameraService())
        source.start()
        fake_capture.gate.set()
        source.stop()

        with pytest.raises(CameraReadError):
            source.read(timeout=0.2)
