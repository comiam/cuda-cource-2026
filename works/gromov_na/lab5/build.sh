#!/bin/bash

BUILD_DIR="build"
if [ ! -d "$BUILD_DIR" ]; then
    mkdir -p "$BUILD_DIR"
fi

cd "$BUILD_DIR"

cmake .. -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
