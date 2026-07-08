from paddleocr import PaddleOCRVL
import os
import tempfile
import time
from datetime import datetime

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_ocr", "paddleocr_vl_1_6")
os.makedirs(output_dir, exist_ok=True)
for existing_file in os.listdir(output_dir):
    existing_path = os.path.join(output_dir, existing_file)
    if os.path.isfile(existing_path):
        os.remove(existing_path)

start_time = time.perf_counter()
print(f"[DEBUG] Started PaddleOCR-VL at {datetime.now().isoformat(timespec='seconds')}", flush=True)
print(f"[DEBUG] Output directory: {output_dir}", flush=True)

print("[DEBUG] Initializing PaddleOCR-VL (first run downloads model weights, can take a while)...", flush=True)
init_start_time = time.perf_counter()
pipeline = PaddleOCRVL(pipeline_version="v1.6")
print(f"[DEBUG] PaddleOCR-VL initialized in {time.perf_counter() - init_start_time:.2f}s", flush=True)

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_mock_docs")
print(f"[DEBUG] Running predict() over: {input_dir}", flush=True)
output = pipeline.predict(input_dir)

all_extracted_text = ""

with tempfile.TemporaryDirectory() as staging_dir:
    for index, res in enumerate(output, start=1):
        result_start_time = time.perf_counter()
        res.save_to_markdown(save_path=staging_dir)
        print(f"[DEBUG] Result {index} processed in {time.perf_counter() - result_start_time:.2f}s", flush=True)

    md_files = sorted(f for f in os.listdir(staging_dir) if f.lower().endswith(".md"))
    for md_file in md_files:
        with open(os.path.join(staging_dir, md_file), encoding="utf-8") as f:
            content = f.read()
        all_extracted_text += f"\n====================\n"
        all_extracted_text += f"FILE: {md_file}\n"
        all_extracted_text += f"====================\n"
        all_extracted_text += content + "\n"

output_file = os.path.join(output_dir, "paddleocr_vl_1_6_output.md")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_extracted_text)

print(f"[DEBUG] Saved extracted text to: {output_file}", flush=True)
print(f"[DEBUG] Finished PaddleOCR-VL at {datetime.now().isoformat(timespec='seconds')}", flush=True)
print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s", flush=True)
