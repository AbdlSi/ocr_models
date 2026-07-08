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

Each model folder is cleared at the start of the run and ends with one ZIP output file:

- `paddleocr_vl_1_6_output.zip`
- `deepseek_ocr_2_output.zip`

## Notes

- `PaddleOCR-VL-1.6` is lighter than DeepSeek-OCR-2 and is the better first test.
- `DeepSeek-OCR-2` is a larger GPU model. If Colab runs out of memory, switch to a higher-memory GPU runtime.
- You can change `INPUT_DIR` and the output folder variables in the configuration cell.
