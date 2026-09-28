#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SAM3_PYTHON:-$ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  printf 'Missing Python environment. See README.md or set SAM3_PYTHON.\n' >&2
  exit 1
fi
export PYTHONNOUSERSITE=1
exec "$PYTHON" "$ROOT/desktop_app.py" "$@"
