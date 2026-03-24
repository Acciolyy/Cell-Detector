#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -d "$ROOT_DIR/.venv" ]]; then
	# shellcheck disable=SC1091
	source "$ROOT_DIR/.venv/bin/activate"
fi

cd "$ROOT_DIR"
python -m src.main --model-path model/cellphone_detector.eim

