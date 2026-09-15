#!/usr/bin/env bash
# Convert MMS TTS ONNX models to RKNN.
# Requires a Linux environment with aarch64 support (or use a Docker image).
# Usage: ./convert_to_rknn.sh [platform] [dtype(optional)]
# Example: ./convert_to_rknn.sh rk3588 fp

set -euo pipefail

# Default arguments
PLATFORM=${1:-rk3588}
DTYPE=${2:-fp}

# Paths
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="$BASE_DIR/../model"
ENCODER_ONNX="$MODEL_DIR/mms_tts_rus_encoder_200.onnx"
DECODER_ONNX="$MODEL_DIR/mms_tts_rus_decoder_200.onnx"

# Create a virtual environment
VENV_DIR="$BASE_DIR/.rknn_venv"
if [ ! -d "$VENV_DIR" ]; then
  python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Install dependencies
# pip install --upgrade pip
# pip install torch==2.4.1 torchaudio==2.4.1 transformers==4.39.3 onnx==1.16.1
# # Install rknn-toolkit2 wheel for aarch64 (Linux)
# # Adjust the wheel path if you have a different version
# WHEEL_PATH="$HOME/rknn-toolkit2/rknn-toolkit2/packages/arm64/rknn_toolkit2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
# if [ -f "$WHEEL_PATH" ]; then
#   pip install "$WHEEL_PATH"
# else
#   echo "Wheel not found: $WHEEL_PATH"
#   exit 1
# fi

# Run conversion for encoder
python3 "$BASE_DIR/convert.py" "$ENCODER_ONNX" "$PLATFORM" "$DTYPE"
# Run conversion for decoder
python3 "$BASE_DIR/convert.py" "$DECODER_ONNX" "$PLATFORM" "$DTYPE"

echo "Conversion finished. RKNN files are in $MODEL_DIR"
