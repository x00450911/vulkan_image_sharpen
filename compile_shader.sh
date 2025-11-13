#!/bin/bash

# Script to compile GLSL compute shader to SPIR-V

if ! command -v glslangValidator &> /dev/null
then
    echo "Error: glslangValidator not found!"
    echo "Please install glslang-tools:"
    echo "  Ubuntu/Debian: sudo apt-get install glslang-tools"
    echo "  Fedora: sudo dnf install glslang"
    echo "  macOS: brew install glslang"
    exit 1
fi

echo "Compiling compute shader..."
glslangValidator -V denoise.comp -o denoise.comp.spv

if [ $? -eq 0 ]; then
    echo "Shader compiled successfully: denoise.comp.spv"
else
    echo "Shader compilation failed!"
    exit 1
fi
