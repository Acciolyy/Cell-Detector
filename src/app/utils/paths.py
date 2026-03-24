"""Path helper utilities."""

from pathlib import Path


def resolve_model_path(model_path: str | Path) -> Path:
	"""Resolve model path to absolute path and validate extension."""
	resolved = Path(model_path).expanduser().resolve()
	if resolved.suffix.lower() != ".eim":
		raise ValueError(f"Expected a .eim file, got: {resolved.name}")
	return resolved


def ensure_file_exists(path: Path) -> None:
	"""Raise descriptive exception when file is missing."""
	if not path.exists() or not path.is_file():
		raise FileNotFoundError(f"File not found: {path}")

