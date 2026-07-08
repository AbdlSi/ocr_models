from paddleocr import PaddleOCR
import os
import time
from datetime import datetime

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_ocr", "paddleocr_pp_ocrv5_multilingual")
os.makedirs(output_dir, exist_ok=True)
for existing_file in os.listdir(output_dir):
    existing_path = os.path.join(output_dir, existing_file)
    if os.path.isfile(existing_path):
        os.remove(existing_path)


def as_dict(res):
    if isinstance(res, dict):
        return res
    for method_name in ("to_dict", "dict", "model_dump"):
        method = getattr(res, method_name, None)
        if callable(method):
            return method()
    try:
        return dict(res)
    except Exception:
        return getattr(res, "__dict__", {})


def extract_text(res):
    data = as_dict(res)
    input_path = data.get("input_path", "unknown")
    page_index = data.get("page_index")
    rec_texts = data.get("rec_texts") or []
    text = "\n".join(str(t) for t in rec_texts)
    return input_path, page_index, text


start_time = time.perf_counter()
print(f"[DEBUG] Started PaddleOCR PP-OCRv5 at {datetime.now().isoformat(timespec='seconds')}", flush=True)
print(f"[DEBUG] Output directory: {output_dir}", flush=True)

print("[DEBUG] Initializing PaddleOCR (first run downloads model weights, can take a while)...", flush=True)
init_start_time = time.perf_counter()
ocr = PaddleOCR(
    lang="tr",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
print(f"[DEBUG] PaddleOCR initialized in {time.perf_counter() - init_start_time:.2f}s", flush=True)

input_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_mock_docs")
print(f"[DEBUG] Running predict() over: {input_dir}", flush=True)
result = ocr.predict(input_dir)

all_extracted_text = ""

for index, res in enumerate(result, start=1):
    result_start_time = time.perf_counter()
    input_path, page_index, text = extract_text(res)
    file_name = os.path.basename(input_path)

    all_extracted_text += f"\n====================\n"
    all_extracted_text += f"FILE: {file_name}\n"
    if page_index is not None:
        all_extracted_text += f"PAGE: {page_index + 1}\n"
    all_extracted_text += f"====================\n"
    all_extracted_text += text + "\n"

    print(
        f"[DEBUG] Result {index} processed in {time.perf_counter() - result_start_time:.2f}s "
        f"({file_name}, page {page_index})",
        flush=True,
    )

output_file = os.path.join(output_dir, "paddleocr_pp_ocrv5_multilingual_output.txt")
with open(output_file, "w", encoding="utf-8") as f:
    f.write(all_extracted_text)

print(f"[DEBUG] Saved extracted text to: {output_file}", flush=True)
print(f"[DEBUG] Finished PaddleOCR PP-OCRv5 at {datetime.now().isoformat(timespec='seconds')}", flush=True)
print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s", flush=True)
