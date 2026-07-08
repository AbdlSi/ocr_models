import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import os
import time
from datetime import datetime

folder_path = r"C:\Users\asus\OneDrive\Desktop\tubitak_sample_docs"
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_ocr", "tesseract_ocr")
os.makedirs(output_dir, exist_ok=True)
for existing_file in os.listdir(output_dir):
    existing_path = os.path.join(output_dir, existing_file)
    if os.path.isfile(existing_path):
        os.remove(existing_path)

start_time = time.perf_counter()
print(f"[DEBUG] Started Tesseract OCR at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Output directory: {output_dir}")

all_extracted_text = ""

for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(".pdf"):
        file_start_time = time.perf_counter()
        pdf_path = os.path.join(folder_path, file_name)

        print(f"Processing: {file_name}")

        pages = convert_from_path(pdf_path, dpi=300)

        all_extracted_text += f"\n====================\n"
        all_extracted_text += f"PDF FILE: {file_name}\n"
        all_extracted_text += f"====================\n"

        for page_number, page_image in enumerate(pages, start=1):
            page_start_time = time.perf_counter()
            text = pytesseract.image_to_string(page_image)

            all_extracted_text += f"\n--- Page {page_number} ---\n"
            all_extracted_text += text + "\n"
            print(f"[DEBUG] Page {page_number} completed in {time.perf_counter() - page_start_time:.2f}s")

        print(f"[DEBUG] File completed in {time.perf_counter() - file_start_time:.2f}s: {file_name}")

print(all_extracted_text)

output_file = os.path.join(output_dir, "tesseract_ocr_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_extracted_text)

print(f"[DEBUG] Saved Tesseract OCR text to: {output_file}")
print(f"[DEBUG] Finished Tesseract OCR at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s")
