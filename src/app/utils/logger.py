"""Central logging configuration helpers."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
	"""Configure the process-wide log format once."""
	logging.basicConfig(
		level=level,
		format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
	)


def get_logger(name: str) -> logging.Logger:
	"""Return a module logger with project formatting."""
	return logging.getLogger(name)

