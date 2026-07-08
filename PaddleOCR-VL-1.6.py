from paddleocr import PaddleOCRVL
import json
import os
import time
from datetime import datetime

output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_ocr", "paddleocr_vl_1_6")
os.makedirs(output_dir, exist_ok=True)
for existing_file in os.listdir(output_dir):
    existing_path = os.path.join(output_dir, existing_file)
    if os.path.isfile(existing_path):
        os.remove(existing_path)


def make_json_safe(value):
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def serialize_result(result):
    if isinstance(result, dict):
        return make_json_safe(result)
    for method_name in ("to_dict", "dict", "model_dump"):
        method = getattr(result, method_name, None)
        if callable(method):
            return make_json_safe(method())
    try:
        return make_json_safe(dict(result))
    except Exception:
        return make_json_safe(getattr(result, "__dict__", str(result)))

start_time = time.perf_counter()
print(f"[DEBUG] Started PaddleOCR-VL at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Output directory: {output_dir}")

pipeline = PaddleOCRVL(batch_size=1,pipeline_version="v1.6" )
output = pipeline.predict(r"C:\Users\asus\OneDrive\Desktop\tubitak_sample_docs")
combined_results = []
for index, res in enumerate(output, start=1):
    result_start_time = time.perf_counter()
    res.print()
    combined_results.append({
        "result_index": index,
        "result": serialize_result(res),
    })
    print(f"[DEBUG] Result {index} processed in {time.perf_counter() - result_start_time:.2f}s")

combined_output_file = os.path.join(output_dir, "paddleocr_vl_1_6_output.json")
with open(combined_output_file, "w", encoding="utf-8") as f:
    json.dump(combined_results, f, ensure_ascii=False, indent=2)

print(f"[DEBUG] Saved combined PaddleOCR-VL output to: {combined_output_file}")
print(f"[DEBUG] Finished PaddleOCR-VL at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s")
