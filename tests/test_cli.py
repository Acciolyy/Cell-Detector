"""Tests for entrypoint argument handling."""

from __future__ import annotations

from pathlib import Path

import pytest

import builtins

from src.app.config.settings import AppSettings
from src.main import _build_arg_parser, _camera_flags_present, main


class TestMissingDependencyMessage:
    def test_error_names_the_missing_module(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "src.app.services.camera_service":
                raise ModuleNotFoundError(
                    "No module named 'edge_impulse_linux'",
                    name="edge_impulse_linux",
                )
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with caplog.at_level("ERROR"):
            with pytest.raises(SystemExit):
                main([])

        # "Missing runtime dependency" alone leaves the user guessing which one.
        assert "edge_impulse_linux" in caplog.text


class TestSourceFlag:
    def test_source_defaults_to_none(self) -> None:
        args = _build_arg_parser().parse_args([])
        assert args.source is None

    def test_source_is_parsed_as_a_path(self) -> None:
        args = _build_arg_parser().parse_args(["--source", "clip.mp4"])
        assert args.source == Path("clip.mp4")

    def test_settings_flag_video_mode(self) -> None:
        assert AppSettings(source_path=Path("clip.mp4")).uses_video_file is True
        assert AppSettings().uses_video_file is False

    def test_headless_file_run_needs_no_record_or_budget(self) -> None:
        # A file ends on its own, so there is a natural stop even without a
        # window; a live camera in the same shape would never terminate.
        AppSettings(source_path=Path("clip.mp4"), show_display=False).validate()

        with pytest.raises(ValueError, match="--no-display"):
            AppSettings(show_display=False).validate()


class TestCameraFlagDetection:
    def test_detects_explicitly_passed_camera_flags(self) -> None:
        argv = ["--source", "clip.mp4", "--camera-width", "1280"]
        assert _camera_flags_present(argv) == ["--camera-width"]

    def test_detects_equals_form(self) -> None:
        assert _camera_flags_present(["--camera-fourcc=MJPG"]) == ["--camera-fourcc"]

    def test_detects_a_default_valued_flag(self) -> None:
        # --camera-index 0 matches the parser default, so comparing against
        # defaults would miss it; argv inspection does not.
        assert _camera_flags_present(["--camera-index", "0"]) == ["--camera-index"]

    def test_reports_nothing_when_no_camera_flags_are_given(self) -> None:
        assert _camera_flags_present(["--source", "clip.mp4", "--no-display"]) == []

    def test_reports_every_camera_flag(self) -> None:
        argv = [
            "--camera-index", "1",
            "--camera-width", "640",
            "--camera-height", "480",
            "--camera-fourcc", "MJPG",
        ]
        assert len(_camera_flags_present(argv)) == 4

    def test_does_not_match_unrelated_flags(self) -> None:
        assert _camera_flags_present(["--record", "out.mp4"]) == []
