#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
exec python3 -B -u "$PROJECT_ROOT/tools/benchmark.py" "$@"
