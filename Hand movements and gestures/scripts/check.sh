#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x "venv/bin/python" ]]; then
  PY="venv/bin/python"
elif [[ -x ".venv/bin/python" ]]; then
  PY=".venv/bin/python"
else
  PY="python3"
fi

echo "==> Compiling sources"
find . -type f -name '*.py' -not -path './venv/*' -not -path './.venv/*' -print0 | xargs -0 "$PY" -m py_compile

echo "==> Running unit tests"
"$PY" -m unittest discover -s tests

echo "==> Smoke test (imports + model load)"
"$PY" scripts/smoke.py

echo "OK"
