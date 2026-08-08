"""Tests for the per-stage timing utilities."""

from __future__ import annotations

import pytest

from src.app.utils.profiling import (
    FrameProfiler,
    MovingAverageFps,
    StageStats,
    percentile,
)


class TestPercentile:
    def test_returns_zero_for_empty_input(self) -> None:
        assert percentile([], 0.95) == 0.0

    def test_uses_nearest_rank(self) -> None:
        samples = list(range(1, 101))  # 1..100
        assert percentile(samples, 0.95) == 95
        assert percentile(samples, 0.5) == 50
        assert percentile(samples, 1.0) == 100

    def test_ignores_input_ordering(self) -> None:
        assert percentile([9.0, 1.0, 5.0], 0.5) == 5.0

    def test_single_sample_is_every_percentile(self) -> None:
        assert percentile([7.5], 0.0) == 7.5
        assert percentile([7.5], 0.95) == 7.5

    def test_rejects_fraction_out_of_range(self) -> None:
        with pytest.raises(ValueError):
            percentile([1.0], 1.5)


class TestStageStats:
    def test_empty_stage_reports_zeroes(self) -> None:
        stats = StageStats("capture")
        assert stats.count == 0
        assert stats.mean_ms() == 0.0
        assert stats.median_ms() == 0.0
        assert stats.p95_ms() == 0.0
        assert stats.last_ms == 0.0

    def test_aggregates_mean_median_and_p95(self) -> None:
        stats = StageStats("inference")
        for value in [10.0, 20.0, 30.0, 40.0]:
            stats.add(value)

        assert stats.count == 4
        assert stats.mean_ms() == pytest.approx(25.0)
        assert stats.median_ms() == pytest.approx(25.0)
        assert stats.p95_ms() == pytest.approx(40.0)
        assert stats.last_ms == pytest.approx(40.0)

    def test_window_keeps_only_recent_samples(self) -> None:
        stats = StageStats("display", window=3)
        for value in [1.0, 2.0, 3.0, 4.0, 5.0]:
            stats.add(value)

        assert stats.count == 3
        assert stats.samples == (3.0, 4.0, 5.0)
        assert stats.mean_ms() == pytest.approx(4.0)

    def test_rejects_negative_duration(self) -> None:
        with pytest.raises(ValueError):
            StageStats("capture").add(-1.0)

    def test_reset_clears_history(self) -> None:
        stats = StageStats("draw")
        stats.add(5.0)
        stats.reset()
        assert stats.count == 0
        assert stats.last_ms == 0.0

    def test_pop_last_undoes_one_sample(self) -> None:
        stats = StageStats("decode")
        stats.add(1.0)
        stats.add(9.0)
        stats.pop_last()

        assert stats.samples == (1.0,)
        assert stats.last_ms == pytest.approx(1.0)

    def test_pop_last_on_empty_stage_is_harmless(self) -> None:
        stats = StageStats("decode")
        stats.pop_last()
        assert stats.count == 0
        assert stats.last_ms == 0.0


class TestFrameProfiler:
    def test_stage_context_records_elapsed_time(self) -> None:
        profiler = FrameProfiler()
        with profiler.stage("capture") as timed:
            pass

        stats = profiler.stats("capture")
        assert stats.count == 1
        assert timed.elapsed_ms >= 0.0
        assert stats.last_ms == pytest.approx(timed.elapsed_ms)

    def test_stage_records_even_when_body_raises(self) -> None:
        profiler = FrameProfiler()
        with pytest.raises(RuntimeError):
            with profiler.stage("inference"):
                raise RuntimeError("boom")

        # A stage that failed still consumed wall-clock time worth reporting.
        assert profiler.stats("inference").count == 1

    def test_same_stage_name_accumulates_into_one_bucket(self) -> None:
        profiler = FrameProfiler()
        for _ in range(3):
            with profiler.stage("draw"):
                pass

        assert profiler.stats("draw").count == 3

    def test_counts_frames_and_computes_effective_fps(self) -> None:
        profiler = FrameProfiler()
        for _ in range(5):
            with profiler.frame_context():
                pass
            profiler.mark_frame_end()

        assert profiler.frames == 5
        assert profiler.wall_elapsed_s > 0.0
        assert profiler.effective_fps() > 0.0

    def test_effective_fps_is_zero_before_any_frame(self) -> None:
        assert FrameProfiler().effective_fps() == 0.0

    def test_record_accepts_externally_measured_durations(self) -> None:
        profiler = FrameProfiler()
        profiler.record("eim_reported", 4.0)
        assert profiler.stats("eim_reported").mean_ms() == pytest.approx(4.0)

    def test_active_stages_follow_loop_order(self) -> None:
        profiler = FrameProfiler()
        for name in ("display", "capture", "inference"):
            profiler.record(name, 1.0)

        assert [stage.name for stage in profiler.active_stages()] == [
            "capture",
            "inference",
            "display",
        ]

    def test_active_stages_skips_empty_buckets(self) -> None:
        profiler = FrameProfiler()
        profiler.stats("capture")  # created but never recorded
        profiler.record("draw", 1.0)

        assert [stage.name for stage in profiler.active_stages()] == ["draw"]

    def test_discard_frame_removes_the_last_iteration(self) -> None:
        profiler = FrameProfiler()
        for _ in range(3):
            with profiler.frame_context():
                profiler.record("decode", 1.0)
            profiler.mark_frame_end()

        # The read that discovers end of file costs time but is not a frame.
        with profiler.frame_context():
            profiler.record("decode", 99.0)
        profiler.discard_frame("decode")

        assert profiler.frames == 3
        assert profiler.frame_total.count == 3
        assert profiler.stats("decode").count == 3
        assert profiler.stats("decode").mean_ms() == pytest.approx(1.0)

    def test_discard_frame_on_empty_profiler_is_harmless(self) -> None:
        profiler = FrameProfiler()
        profiler.discard_frame("decode")
        assert profiler.frames == 0

    def test_report_names_the_source_when_given(self) -> None:
        profiler = FrameProfiler()
        with profiler.frame_context():
            profiler.record("decode", 1.0)
        profiler.mark_frame_end()

        assert "source: video file clip.mp4" in profiler.format_report(
            source="video file clip.mp4"
        )

    def test_report_lists_every_recorded_stage(self) -> None:
        profiler = FrameProfiler()
        with profiler.frame_context():
            profiler.record("capture", 100.0)
            profiler.record("inference", 5.0)
        profiler.mark_frame_end()

        report = profiler.format_report("Baseline")
        assert "Baseline" in report
        assert "capture" in report
        assert "inference" in report
        assert "frame total" in report
        assert "effective FPS" in report


class TestMovingAverageFps:
    def test_zero_before_the_second_frame(self) -> None:
        meter = MovingAverageFps(window=30)
        assert meter.value() == 0.0
        assert meter.tick(now=0.0) == 0.0

    def test_averages_the_interval_not_the_rate(self) -> None:
        meter = MovingAverageFps(window=30)
        # Ten frames, 20 ms apart -> 50 FPS.
        for index in range(11):
            meter.tick(now=index * 0.02)

        assert meter.value() == pytest.approx(50.0)

    def test_smooths_a_single_slow_frame(self) -> None:
        meter = MovingAverageFps(window=10)
        for index in range(10):
            meter.tick(now=index * 0.01)  # 100 FPS
        meter.tick(now=0.09 + 0.5)  # one 500 ms stall

        # Instantaneous 1/dt would read 2 FPS; the average stays informative.
        assert 10.0 < meter.value() < 30.0

    def test_window_bounds_the_history(self) -> None:
        meter = MovingAverageFps(window=2)
        meter.tick(now=0.0)
        meter.tick(now=1.0)  # 1 s interval, should fall out of the window
        meter.tick(now=1.1)
        meter.tick(now=1.2)

        assert meter.value() == pytest.approx(10.0)

    def test_ignores_non_advancing_clock(self) -> None:
        meter = MovingAverageFps(window=5)
        meter.tick(now=1.0)
        meter.tick(now=1.0)
        assert meter.value() == 0.0

    def test_reset_clears_history(self) -> None:
        meter = MovingAverageFps(window=5)
        meter.tick(now=0.0)
        meter.tick(now=0.1)
        meter.reset()
        assert meter.value() == 0.0

    def test_rejects_invalid_window(self) -> None:
        with pytest.raises(ValueError):
            MovingAverageFps(window=0)
