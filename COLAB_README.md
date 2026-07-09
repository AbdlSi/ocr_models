# Running the OCR models with VS Code + Google Colab

Use `colab_ocr_models.ipynb` from VS Code with your Google Colab extension.

## Setup

1. Upload or sync your input folder to Google Drive:
   `MyDrive/tubitak_sample_docs`

2. Open `colab_ocr_models.ipynb` in VS Code.

3. Connect the notebook to a Google Colab GPU runtime.

4. Run the cells from top to bottom.

The notebook writes all model outputs under:

`/content/drive/MyDrive/output/output_ocr`

Each model has a separate folder:

- `/content/drive/MyDrive/output/output_ocr/paddleocr_vl_1_6`
- `/content/drive/MyDrive/output/output_ocr/deepseek_ocr_2`

Each model folder is cleared at the start of the run and ends with one text-only Markdown output file:

- `paddleocr_vl_1_6_output.md`
- `deepseek_ocr_2_output.md`

## Notes

- Use a Python 3 GPU runtime such as T4/L4/A100. These model paths use CUDA and are not TPU-compatible.
- `PaddleOCR-VL-1.6` is lighter than DeepSeek-OCR-2 and is the better first test.
- `DeepSeek-OCR-2` is a larger GPU model. The notebook uses a low-VRAM preset by default: lower PDF DPI, smaller inference size, no crop tiling, text-only results, page-by-page processing, and CUDA cache cleanup after each page.
- If DeepSeek output quality is too low and your GPU has enough VRAM, set `DEEPSEEK_LOW_VRAM = False` in the DeepSeek cell.
- You can change `INPUT_DIR` and the output folder variables in the configuration cell.
