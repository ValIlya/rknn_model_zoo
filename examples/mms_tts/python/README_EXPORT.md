# MMS TTS ONNX Export

This repository contains a small helper script to export the MMS TTS example from the
[RKNN Model Zoo](https://github.com/rknnlite/rknn_model_zoo) to ONNX format.

## Prerequisites

* Python 3.11+ (tested on 3.11.15)
* `uv` package manager (used for fast dependency installation)

## Steps

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/rknnlite/rknn_model_zoo.git
   cd rknn_model_zoo/examples/mms_tts/python
   ```

2. **Run the export script**:
   ```bash
   ./run_export.sh 200   # 200 is the max_length argument
   ```
   The script will:
   * Create a virtual environment in `.venv` if it doesn't exist.
   * Install the required Python packages: `torch`, `torchaudio`, `transformers`, `onnx`.
   * Patch the local `modeling_vits_for_export_onnx.py` to use absolute imports from the
     installed `transformers` package.
   * Execute `export_onnx.py` with the specified `max_length`.

3. **Result**
   Two ONNX files will be generated in the `../model` directory:
   * `mms_tts_eng_encoder_200.onnx`
   * `mms_tts_eng_decoder_200.onnx`

4. **Convert to RKNN** (optional)
   Use the provided `convert.py` script to convert the ONNX files to RKNN format:
   ```bash
   python3 convert.py --encoder ../model/mms_tts_eng_encoder_200.onnx \
                     --decoder ../model/mms_tts_eng_decoder_200.onnx \
                     --output_dir ../model
   ```

## Notes

* The script uses `opset_version=17` for ONNX export to avoid compatibility issues.
* If you encounter import errors, ensure that the `transformers` package is installed in the
  same virtual environment and that the patching step was successful.
* The script is idempotent: running it multiple times will reuse the existing virtual
  environment and only reinstall packages if necessary.
