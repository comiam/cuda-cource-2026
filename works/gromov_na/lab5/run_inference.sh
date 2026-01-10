#!/bin/bash

if [ $# -lt 2 ]; then
    echo "Использование: $0 <engine_file> <input_image_or_video> [output_path] [confidence_threshold]"
    echo "Примеры:"
    echo "  $0 models/retinanet_int8_raw.trt input/test_image.jpg output/result.jpg"
    echo "  $0 models/retinanet_int8_raw.trt input/test_video.mp4 output/result.mp4 0.3"
    exit 1
fi

ENGINE_FILE=$1
INPUT_FILE=$2
OUTPUT_PATH=${3:-""}
CONF_THRESHOLD=${4:-"0.1"}
LABELS_FILE=$(dirname "$ENGINE_FILE")/labels.txt

./build/retinanet_tensorrt "$ENGINE_FILE" "$INPUT_FILE" "$OUTPUT_PATH" "$CONF_THRESHOLD"