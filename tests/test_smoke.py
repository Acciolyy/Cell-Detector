"""Smoke tests for basic project integrity."""

from pathlib import Path

from src.app.config.settings import AppSettings
from src.main import _build_arg_parser


def test_arg_parser_defaults() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args([])

    assert str(args.model_path).endswith("cellphone_detector.eim")
    assert args.camera_index == 0
    assert 0.0 <= args.min_confidence <= 1.0


def test_settings_validation_accepts_valid_values() -> None:
    settings = AppSettings(model_path=Path("model/cellphone_detector.eim"), min_confidence=0.5)
    settings.validate()

