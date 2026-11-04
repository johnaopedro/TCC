#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$PROJECT_ROOT"
for dependency in g++ pkg-config; do
  command -v "$dependency" >/dev/null || { echo "ERRO: dependencia ausente: $dependency" >&2; exit 2; }
done
read -r -a CV_FLAGS <<< "$(pkg-config --cflags --libs opencv4)"
FLAGS=(-O3 -std=c++17 -ffp-contract=off -Wno-unknown-pragmas)
mkdir -p bin/sequenciais bin/paralelos
for algorithm in threshold equalization sobel-prewitt-roberts; do
  echo "Compilando $algorithm (referencia sequencial e versao paralelizada)..."
  g++ "${FLAGS[@]}" "originais/$algorithm.cpp" "${CV_FLAGS[@]}" -o "bin/sequenciais/$algorithm"
  g++ "${FLAGS[@]}" -fopenmp "paralelizados/$algorithm.cpp" "${CV_FLAGS[@]}" -o "bin/paralelos/$algorithm"
done
echo "OK: seis executaveis em bin/{sequenciais,paralelos}/"
