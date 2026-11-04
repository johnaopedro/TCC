#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
for dependency in g++ pkg-config; do
  command -v "$dependency" >/dev/null || { echo "ERRO: dependencia ausente: $dependency" >&2; exit 2; }
done
read -r -a CV_FLAGS <<< "$(pkg-config --cflags --libs opencv4)"
TEST_BUILD_DIR=$(mktemp -d "${TMPDIR:-/tmp}/tcc-verificacao.XXXXXX")
trap 'rm -r -- "$TEST_BUILD_DIR"' EXIT
g++ -O3 -std=c++17 -ffp-contract=off -fopenmp -DEXP_PAR=1 \
  testes/verify.cpp "${CV_FLAGS[@]}" -o "$TEST_BUILD_DIR/verificar"
echo "Verificando 44 entradas com 1, 2, 4, 6 e 8 threads..."
OMP_DYNAMIC=false OMP_PLACES=cores OMP_PROC_BIND=close OMP_WAIT_POLICY=PASSIVE "$TEST_BUILD_DIR/verificar"
