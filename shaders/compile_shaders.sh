#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shaders=(
  "rotation_sharpen.comp"
  "exposure_reduce.comp"
  "denoise.comp"
  "super_resolution.comp"
)

for shader in "${shaders[@]}"; do
  glslangValidator -V "${root_dir}/${shader}" -o "${root_dir}/${shader}.spv"
done
