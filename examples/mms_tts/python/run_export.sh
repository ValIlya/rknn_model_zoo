#!/usr/bin/env bash
# Script to export the MMS TTS example to ONNX.
# Usage: ./run_export.sh [max_length]
# Example: ./run_export.sh 200

set -euo pipefail

# Default max length
MAX_LENGTH=${1:-200}

# Directory structure
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$BASE_DIR/.venv"
MODEL_DIR="$BASE_DIR/../model"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV_DIR"
  source "$VENV_DIR/bin/activate"
  echo "Installing dependencies..."
  uv pip install --upgrade pip
  uv pip install torch==2.4.1 torchaudio==2.4.1 transformers==4.39.3 onnx==1.15.0
else
  source "$VENV_DIR/bin/activate"
fi

# Patch the custom VITS model to use absolute imports
PATCH_FILE="$BASE_DIR/modeling_vits_for_export_onnx.py"
if grep -q "from ...activations" "$PATCH_FILE"; then
  echo "Patching imports in $PATCH_FILE..."
  sed -i.bak -e 's/from ...activations/from transformers.activations/' \
      -e 's/from ...integrations.deepspeed/from transformers.integrations.deepspeed/' \
      -e 's/from ...modeling_attn_mask_utils/from transformers.modeling_attn_mask_utils/' \
      -e 's/from ...modeling_outputs/from transformers.modeling_outputs/' \
      -e 's/from ...modeling_utils/from transformers.modeling_utils/' \
      -e 's/from ...utils/from transformers.utils/' \
      -e 's/from .configuration_vits/from transformers.models.vits.configuration_vits/' "$PATCH_FILE"
fi

# Run the export script
echo "Running export_onnx.py with max_length=$MAX_LENGTH..."
uv run export_onnx.py --max_length "$MAX_LENGTH"

echo "Export completed. ONNX files are in $MODEL_DIR"
