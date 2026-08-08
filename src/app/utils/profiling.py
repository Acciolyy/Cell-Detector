"""Per-stage timing utilities for the real-time detection loop.

The pipeline is a chain of stages (capture, preprocess, inference, draw,
display). A single end-to-end latency number hides which one is the
bottleneck, so every stage is timed independently and aggregated here.
"""

from __future__ import annotations

import math
import statistics
import time
from collections import deque
from types import TracebackType
from typing import Iterator, Sequence

# Canonical stage names, in loop execution order.
STAGE_CAPTURE = "capture"
# Reading a file is disk and codec work, not camera work; keeping it under a
# separate name stops the two from being averaged into one meaningless number.
STAGE_DECODE = "decode"
STAGE_PREPROCESS = "preprocess"
STAGE_INFERENCE = "inference"
STAGE_DRAW = "draw"
STAGE_DISPLAY = "display"
STAGE_RECORD = "record"

STAGE_ORDER: tuple[str, ...] = (
    STAGE_CAPTURE,
    STAGE_DECODE,
    STAGE_PREPROCESS,
    STAGE_INFERENCE,
    STAGE_DRAW,
    STAGE_DISPLAY,
    STAGE_RECORD,
)


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of ``samples`` (``fraction`` in [0.0, 1.0])."""
    if not samples:
        return 0.0
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0.0 and 1.0")

    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


class StageStats:
    """Rolling collection of durations (ms) recorded for a single stage."""

    __slots__ = ("_name", "_samples", "_last_ms")

    def __init__(self, name: str, window: int | None = None) -> None:
        self._name = name
        self._samples: deque[float] = deque(maxlen=window)
        self._last_ms: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def count(self) -> int:
        return len(self._samples)

    @property
    def last_ms(self) -> float:
        """Duration of the most recent sample, or 0.0 when empty."""
        return self._last_ms

    @property
    def samples(self) -> tuple[float, ...]:
        return tuple(self._samples)

    def add(self, duration_ms: float) -> None:
        """Record one duration sample in milliseconds."""
        if duration_ms < 0.0:
            raise ValueError("duration_ms must be >= 0.0")
        self._samples.append(duration_ms)
        self._last_ms = duration_ms

    def mean_ms(self) -> float:
        return statistics.fmean(self._samples) if self._samples else 0.0

    def median_ms(self) -> float:
        return statistics.median(self._samples) if self._samples else 0.0

    def p95_ms(self) -> float:
        return percentile(self._samples, 0.95)

    def pop_last(self) -> None:
        """Remove the most recent sample.

        Used to undo the timing of an iteration that turned out not to be a
        frame at all, such as the read that discovers end of file.
        """
        if self._samples:
            self._samples.pop()
        self._last_ms = self._samples[-1] if self._samples else 0.0

    def reset(self) -> None:
        self._samples.clear()
        self._last_ms = 0.0


class _StageContext:
    """Context manager that records its own wall-clock duration into a stage."""

    __slots__ = ("_stats", "_started_at", "elapsed_ms")

    def __init__(self, stats: StageStats) -> None:
        self._stats = stats
        self._started_at = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self) -> "_StageContext":
        self._started_at = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        # Record even on failure: a stage that raises still consumed time.
        self.elapsed_ms = (time.perf_counter() - self._started_at) * 1000.0
        self._stats.add(self.elapsed_ms)


class FrameProfiler:
    """Aggregates per-stage timings plus total frame time for the loop.

    ``window`` bounds how many samples each stage keeps. ``None`` keeps all
    samples, which is what the fixed-length profiling run uses.
    """

    def __init__(self, window: int | None = None) -> None:
        self._window = window
        self._stages: dict[str, StageStats] = {}
        self._frame_total = StageStats("frame", window=window)
        self._frames = 0
        self._wall_started_at: float | None = None
        self._wall_elapsed_s = 0.0

    @property
    def frames(self) -> int:
        return self._frames

    @property
    def frame_total(self) -> StageStats:
        return self._frame_total

    @property
    def wall_elapsed_s(self) -> float:
        """Wall-clock seconds spanned by the recorded frames."""
        if self._wall_started_at is None:
            return 0.0
        return self._wall_elapsed_s

    def stats(self, name: str) -> StageStats:
        """Return (creating on first use) the stats bucket for ``name``."""
        stage = self._stages.get(name)
        if stage is None:
            stage = StageStats(name, window=self._window)
            self._stages[name] = stage
        return stage

    def stage(self, name: str) -> _StageContext:
        """Time a block of work and attribute it to stage ``name``."""
        return _StageContext(self.stats(name))

    def record(self, name: str, duration_ms: float) -> None:
        """Record a duration measured elsewhere (e.g. reported by the runner)."""
        self.stats(name).add(duration_ms)

    def frame_context(self) -> _StageContext:
        """Time one whole loop iteration."""
        if self._wall_started_at is None:
            self._wall_started_at = time.perf_counter()
        self._frames += 1
        return _StageContext(self._frame_total)

    def discard_frame(self, *stage_names: str) -> None:
        """Undo the last frame, and the named stages, from the statistics.

        The iteration that discovers end of stream costs time but produces no
        frame; counting it would understate every stage's share and skew the
        effective FPS.
        """
        if self._frames > 0:
            self._frames -= 1
        self._frame_total.pop_last()
        for name in stage_names:
            self.stats(name).pop_last()

    def mark_frame_end(self) -> None:
        """Update wall-clock span; call once per frame after ``frame_context``."""
        if self._wall_started_at is not None:
            self._wall_elapsed_s = time.perf_counter() - self._wall_started_at

    def effective_fps(self) -> float:
        """Frames per second over the whole profiling window."""
        if self._frames <= 0 or self.wall_elapsed_s <= 0.0:
            return 0.0
        return self._frames / self.wall_elapsed_s

    def active_stages(self) -> Iterator[StageStats]:
        """Yield recorded stages in canonical loop order, extras last."""
        for name in STAGE_ORDER:
            stage = self._stages.get(name)
            if stage is not None and stage.count:
                yield stage
        for name, stage in self._stages.items():
            if name not in STAGE_ORDER and stage.count:
                yield stage

    def format_report(
        self, title: str = "Stage timing summary", source: str | None = None
    ) -> str:
        """Render a fixed-width table with mean/median/p95 per stage.

        ``source`` is printed verbatim so a report is never ambiguous about
        whether the frames came from a camera or from a file.
        """
        header = (
            f"{'stage':<14}{'n':>6}{'mean ms':>10}"
            f"{'median ms':>12}{'p95 ms':>10}{'share':>9}"
        )
        lines = [title]
        if source is not None:
            lines.append(f"source: {source}")
        lines += ["=" * len(header), header, "-" * len(header)]

        total_mean = self._frame_total.mean_ms()
        for stage in self.active_stages():
            mean = stage.mean_ms()
            share = (mean / total_mean * 100.0) if total_mean > 0.0 else 0.0
            lines.append(
                f"{stage.name:<14}{stage.count:>6}{mean:>10.2f}"
                f"{stage.median_ms():>12.2f}{stage.p95_ms():>10.2f}{share:>8.1f}%"
            )

        lines.append("-" * len(header))
        lines.append(
            f"{'frame total':<14}{self._frame_total.count:>6}{total_mean:>10.2f}"
            f"{self._frame_total.median_ms():>12.2f}"
            f"{self._frame_total.p95_ms():>10.2f}{100.0:>8.1f}%"
        )
        lines.append("=" * len(header))
        lines.append(
            f"frames={self._frames}  wall={self.wall_elapsed_s:.2f}s  "
            f"effective FPS={self.effective_fps():.2f}"
        )
        return "\n".join(lines)


class MovingAverageFps:
    """FPS smoothed over the last ``window`` frame intervals.

    Instantaneous ``1/dt`` is far too noisy to display; this keeps the HUD
    number readable while still reacting within a second.
    """

    def __init__(self, window: int = 30) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._intervals: deque[float] = deque(maxlen=window)
        self._last_tick: float | None = None

    @property
    def window(self) -> int:
        return self._intervals.maxlen or 0

    def tick(self, now: float | None = None) -> float:
        """Register a frame boundary and return the smoothed FPS."""
        current = time.perf_counter() if now is None else now
        if self._last_tick is not None:
            delta = current - self._last_tick
            if delta > 0.0:
                self._intervals.append(delta)
        self._last_tick = current
        return self.value()

    def value(self) -> float:
        """Current smoothed FPS, or 0.0 before enough samples exist."""
        if not self._intervals:
            return 0.0
        average = statistics.fmean(self._intervals)
        return 1.0 / average if average > 0.0 else 0.0

    def reset(self) -> None:
        self._intervals.clear()
        self._last_tick = None
