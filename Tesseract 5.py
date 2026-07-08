import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import os
import time
from datetime import datetime

folder_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_mock_docs")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_ocr", "tesseract_ocr")
os.makedirs(output_dir, exist_ok=True)
for existing_file in os.listdir(output_dir):
    existing_path = os.path.join(output_dir, existing_file)
    if os.path.isfile(existing_path):
        os.remove(existing_path)

PDF_EXTENSIONS = (".pdf",)
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
MAX_WORKERS = min(8, os.cpu_count() or 4)

start_time = time.perf_counter()
print(f"[DEBUG] Started Tesseract OCR at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Output directory: {output_dir}")
print(f"[DEBUG] Using {MAX_WORKERS} worker threads")

all_extracted_text = ""


def ocr_page(page_number, page_image):
    page_start_time = time.perf_counter()
    text = pytesseract.image_to_string(page_image)
    elapsed = time.perf_counter() - page_start_time
    return page_number, text, elapsed


file_names = sorted(
    f for f in os.listdir(folder_path)
    if f.lower().endswith(PDF_EXTENSIONS + IMAGE_EXTENSIONS)
)

for file_name in file_names:
    file_start_time = time.perf_counter()
    file_path = os.path.join(folder_path, file_name)

    print(f"Processing: {file_name}")

    if file_name.lower().endswith(PDF_EXTENSIONS):
        pages = convert_from_path(file_path, dpi=300)
    else:
        pages = [Image.open(file_path)]

    all_extracted_text += f"\n====================\n"
    all_extracted_text += f"FILE: {file_name}\n"
    all_extracted_text += f"====================\n"

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(
            lambda item: ocr_page(item[0], item[1]),
            enumerate(pages, start=1),
        ))

    for page_number, text, elapsed in results:
        all_extracted_text += f"\n--- Page {page_number} ---\n"
        all_extracted_text += text + "\n"
        print(f"[DEBUG] Page {page_number} completed in {elapsed:.2f}s")

    print(f"[DEBUG] File completed in {time.perf_counter() - file_start_time:.2f}s: {file_name}")

print(all_extracted_text)

output_file = os.path.join(output_dir, "tesseract_ocr_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_extracted_text)

print(f"[DEBUG] Saved Tesseract OCR text to: {output_file}")
print(f"[DEBUG] Finished Tesseract OCR at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s")
