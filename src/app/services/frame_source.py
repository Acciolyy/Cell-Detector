"""Frame producers for the detection pipeline.

Two sources, deliberately kept as separate implementations rather than one
class with mode flags, because they differ in the one place that matters:

- :class:`ThreadedFrameSource` (live webcam) keeps only the newest frame. In
  real time a frame the consumer was too slow to take is worthless, and
  queueing it would only add latency.
- :class:`VideoFileFrameSource` (recorded file) delivers every frame in order.
  A dropped frame there is lost content, not stale content.
"""

from __future__ import annotations

import queue
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from types import TracebackType
from typing import Any

import cv2
import numpy as np

from src.app.services.camera_service import CameraReadError, CameraService
from src.app.utils.logger import get_logger
from src.app.utils.profiling import STAGE_CAPTURE, STAGE_DECODE

LOGGER = get_logger(__name__)

# Transient read failures happen on USB re-enumeration; only give up when
# they persist, so a single glitch does not kill a running demo.
MAX_CONSECUTIVE_READ_FAILURES = 30

# Depth of the decode queue. Enough to overlap decoding with inference,
# small enough that stopping does not have to discard much work.
DECODE_QUEUE_DEPTH = 8

# Frame rate assumed when a container reports a nonsensical one.
FALLBACK_VIDEO_FPS = 30.0
MAX_PLAUSIBLE_VIDEO_FPS = 1000.0


class EndOfStream(Exception):
    """Raised when a finite source has no frames left.

    Not an error: reaching the end of a video file is the normal way a
    file-backed run finishes.
    """


class VideoSourceError(RuntimeError):
    """Raised when a video file cannot be used as a frame source."""


class FrameSource(ABC):
    """Common interface the pipeline consumes frames through."""

    #: Profiler stage name, so file decoding is not reported as camera capture.
    stage_name: str = STAGE_CAPTURE

    @abstractmethod
    def start(self) -> "FrameSource":
        """Acquire the underlying device or file and begin producing."""

    @abstractmethod
    def read(self, timeout: float = 5.0) -> np.ndarray:
        """Return the next frame, or raise :class:`EndOfStream` when finished."""

    @abstractmethod
    def stop(self) -> None:
        """Release every resource. Safe to call twice, and from a finally."""

    @property
    @abstractmethod
    def frames_read(self) -> int:
        """Frames pulled from the underlying source so far."""

    @property
    def frames_dropped(self) -> int:
        """Frames discarded before the consumer could take them."""
        return 0

    @property
    def output_fps(self) -> float | None:
        """Frame rate a recording of this source should use.

        ``None`` means the source has no inherent rate and the measured
        processing rate should be used instead.
        """
        return None

    @abstractmethod
    def describe(self) -> str:
        """Short human-readable summary for logs and the timing report."""

    def __enter__(self) -> "FrameSource":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.stop()


class ThreadedFrameSource(FrameSource):
    """Continuously reads webcam frames in the background, keeping the newest.

    ``cv2.VideoCapture.read()`` is a blocking dequeue: the calling thread
    sleeps until the driver hands over the next frame. Running it inline makes
    the whole pipeline advance at the camera's period, no matter how fast
    inference is. Moving it to a producer thread decouples the two, and
    dropping rather than queueing keeps the displayed frame current.
    """

    stage_name = STAGE_CAPTURE

    def __init__(
        self,
        camera_service: CameraService,
        max_read_failures: int = MAX_CONSECUTIVE_READ_FAILURES,
        thread_name: str = "frame-producer",
    ) -> None:
        self._camera_service = camera_service
        self._max_read_failures = max_read_failures
        self._thread_name = thread_name

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._condition = threading.Condition()

        self._slot: np.ndarray | None = None
        self._frames_captured = 0
        self._frames_dropped = 0
        self._error: BaseException | None = None

    @property
    def frames_read(self) -> int:
        return self._frames_captured

    @property
    def frames_captured(self) -> int:
        """Frames pulled from the driver since start."""
        return self._frames_captured

    @property
    def frames_dropped(self) -> int:
        """Frames overwritten before the consumer could read them."""
        return self._frames_dropped

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def describe(self) -> str:
        return f"webcam index {self._camera_service.camera_index}"

    def start(self) -> "ThreadedFrameSource":
        """Open the camera and start the producer thread."""
        if self._thread is not None:
            raise RuntimeError("ThreadedFrameSource is already started.")

        self._camera_service.open()
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._produce, name=self._thread_name, daemon=True
        )
        self._thread.start()
        return self

    def read(self, timeout: float = 5.0) -> np.ndarray:
        """Return the newest frame, waiting for one if the slot is empty.

        Once a frame is taken the slot is cleared, so the consumer always sees
        a frame it has not processed yet.
        """
        with self._condition:
            if self._slot is None:
                self._condition.wait_for(
                    lambda: self._slot is not None
                    or self._error is not None
                    or self._stop_event.is_set(),
                    timeout=timeout,
                )

            if self._slot is not None:
                frame = self._slot
                self._slot = None
                return frame

            if self._error is not None:
                raise CameraReadError(str(self._error)) from self._error

            if self._stop_event.is_set():
                raise CameraReadError("Frame source was stopped.")

        raise CameraReadError(
            f"No frame received from the camera within {timeout:.1f}s."
        )

    def stop(self) -> None:
        """Stop the producer thread and release the camera.

        Safe to call more than once and from an exception handler; every step
        runs even if an earlier one fails.
        """
        self._stop_event.set()
        with self._condition:
            # Wake a consumer blocked in read() so shutdown cannot deadlock.
            self._condition.notify_all()

        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
            if thread.is_alive():  # pragma: no cover - driver wedged in read()
                LOGGER.warning(
                    "Frame producer thread did not stop within timeout; "
                    "it is a daemon and will not block process exit."
                )

        self._camera_service.release()

        with self._condition:
            self._slot = None

    def __enter__(self) -> "ThreadedFrameSource":
        return self.start()

    def _produce(self) -> None:
        """Producer loop: read, replace the slot, drop whatever was there."""
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                frame = self._camera_service.read_frame()
            except CameraReadError as read_error:
                consecutive_failures += 1
                if consecutive_failures >= self._max_read_failures:
                    LOGGER.error(
                        "Camera failed %s consecutive reads; stopping producer.",
                        consecutive_failures,
                    )
                    self._fail(read_error)
                    return
                continue
            except BaseException as unexpected:  # noqa: BLE001 - must not die silently
                self._fail(unexpected)
                return

            consecutive_failures = 0
            with self._condition:
                if self._slot is not None:
                    # Consumer is slower than the camera: keep only the newest.
                    self._frames_dropped += 1
                self._slot = frame
                self._frames_captured += 1
                self._condition.notify()

    def _fail(self, error: BaseException) -> None:
        """Publish a producer-side failure to whoever calls read() next."""
        with self._condition:
            self._error = error
            self._condition.notify_all()


class VideoFileFrameSource(FrameSource):
    """Decodes a video file, delivering every frame in order.

    Decoding runs in a producer thread to overlap with inference, but through
    a bounded blocking queue rather than a single slot: nothing is dropped. If
    the consumer falls behind, the decoder waits instead of skipping ahead.
    """

    stage_name = STAGE_DECODE

    def __init__(
        self,
        path: Path,
        queue_depth: int = DECODE_QUEUE_DEPTH,
        thread_name: str = "decode-producer",
    ) -> None:
        self._path = path
        self._thread_name = thread_name

        self._capture: cv2.VideoCapture | None = None
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(
            maxsize=max(1, queue_depth)
        )
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        self._frames_read = 0
        self._source_fps: float | None = None
        self._frame_count: int | None = None
        self._error: BaseException | None = None
        self._exhausted = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def frames_read(self) -> int:
        return self._frames_read

    @property
    def output_fps(self) -> float | None:
        """Frame rate of the input file.

        A recording made from a file must inherit this, not the rate at which
        the machine happened to process it, or playback comes out fast or slow.
        """
        return self._source_fps

    @property
    def frame_count(self) -> int | None:
        """Frame count reported by the container, when it reports one."""
        return self._frame_count

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def describe(self) -> str:
        total = f"{self._frame_count} frames" if self._frame_count else "length unknown"
        fps = f"{self._source_fps:.2f} fps" if self._source_fps else "fps unknown"
        return f"video file {self._path.name} ({total}, {fps})"

    def start(self) -> "VideoFileFrameSource":
        """Validate and open the file, then start the decoder thread."""
        if self._thread is not None:
            raise RuntimeError("VideoFileFrameSource is already started.")

        self._validate_path()

        capture = cv2.VideoCapture(str(self._path))
        if not capture.isOpened():
            capture.release()
            raise VideoSourceError(
                f"Could not open video file: {self._path}. The container or "
                f"codec may not be supported by this OpenCV build."
            )

        # Probe the first frame now so an unreadable file fails at startup with
        # a clear message, rather than mid-run. The probed frame is kept and
        # handed to the consumer first, so nothing is consumed by the check.
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            capture.release()
            raise VideoSourceError(
                f"Video file contains no readable frames: {self._path}"
            )

        self._capture = capture
        self._source_fps = _resolve_video_fps(capture.get(cv2.CAP_PROP_FPS), self._path)
        self._frame_count = _resolve_frame_count(capture.get(cv2.CAP_PROP_FRAME_COUNT))

        self._queue.put(first_frame)
        self._frames_read = 1

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._decode, name=self._thread_name, daemon=True
        )
        self._thread.start()

        LOGGER.info("Reading frames from %s", self.describe())
        return self

    def read(self, timeout: float = 5.0) -> np.ndarray:
        """Return the next frame in order, raising EndOfStream when done."""
        if self._exhausted:
            raise EndOfStream(f"End of video file: {self._path}")

        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty as empty:
            raise VideoSourceError(
                f"Decoder produced no frame within {timeout:.1f}s for {self._path}."
            ) from empty

        if item is None:
            # Sentinel: the decoder finished or failed.
            self._exhausted = True
            if self._error is not None:
                raise VideoSourceError(
                    f"Failed while decoding {self._path}: {self._error}"
                ) from self._error
            raise EndOfStream(f"End of video file: {self._path}")

        return item

    def stop(self) -> None:
        """Stop the decoder thread and release the file handle."""
        self._stop_event.set()

        thread = self._thread
        self._thread = None
        if thread is not None and thread.is_alive():
            # The decoder may be parked on a full queue; drain it so the
            # thread can observe the stop flag and exit.
            while thread.is_alive():
                _drain(self._queue)
                thread.join(timeout=0.1)

        if self._capture is not None:
            self._capture.release()
            self._capture = None

        _drain(self._queue)

    def __enter__(self) -> "VideoFileFrameSource":
        return self.start()

    def _validate_path(self) -> None:
        """Fail early with a message that names the actual problem."""
        if self._path.is_dir():
            raise VideoSourceError(
                f"Video source is a directory, not a file: {self._path}"
            )
        if not self._path.exists():
            raise VideoSourceError(f"Video file not found: {self._path}")
        if not self._path.is_file():
            raise VideoSourceError(
                f"Video source is not a regular file: {self._path}"
            )

    def _decode(self) -> None:
        """Decoder loop: every frame, in order, no drops."""
        capture = self._capture
        if capture is None:  # pragma: no cover - start() guarantees this
            self._finish()
            return

        while not self._stop_event.is_set():
            try:
                ok, frame = capture.read()
            except BaseException as unexpected:  # noqa: BLE001
                self._error = unexpected
                break

            if not ok or frame is None:
                break  # Normal end of file.

            if not self._offer(frame):
                return  # Stopped while waiting for queue space.

            self._frames_read += 1

        self._finish()

    def _offer(self, frame: np.ndarray) -> bool:
        """Block until the queue has room, unless we are shutting down."""
        while not self._stop_event.is_set():
            try:
                self._queue.put(frame, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _finish(self) -> None:
        """Push the end-of-stream sentinel, waiting for room if needed."""
        while not self._stop_event.is_set():
            try:
                self._queue.put(None, timeout=0.1)
                return
            except queue.Full:
                continue


def _drain(target: "queue.Queue[Any]") -> None:
    """Empty a queue without blocking."""
    while True:
        try:
            target.get_nowait()
        except queue.Empty:
            return


def _resolve_video_fps(raw_fps: float, path: Path) -> float:
    """Return a usable frame rate, falling back when the container lies."""
    try:
        fps = float(raw_fps)
    except (TypeError, ValueError):
        fps = 0.0

    # NaN fails every comparison, so test it by identity with itself.
    if fps != fps or fps <= 0.0 or fps > MAX_PLAUSIBLE_VIDEO_FPS:
        LOGGER.warning(
            "Video file %s reports an unusable frame rate (%r); "
            "falling back to %.1f fps.",
            path,
            raw_fps,
            FALLBACK_VIDEO_FPS,
        )
        return FALLBACK_VIDEO_FPS

    return fps


def _resolve_frame_count(raw_count: float) -> int | None:
    """Return the container's frame count when it reports a plausible one."""
    try:
        count = int(raw_count)
    except (TypeError, ValueError, OverflowError):
        return None
    return count if count > 0 else None


def describe_source(source: Any) -> str:
    """Short human-readable capture summary for logs and reports."""
    read = getattr(source, "frames_read", 0)
    dropped = getattr(source, "frames_dropped", 0)
    total = read or 1
    return f"read={read} dropped={dropped} ({dropped / total * 100.0:.1f}%)"
