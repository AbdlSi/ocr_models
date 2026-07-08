from transformers import AutoModel, AutoTokenizer
import torch
import os
import time
from datetime import datetime


model_name = 'deepseek-ai/DeepSeek-OCR-2'

start_time = time.perf_counter()
print(f"[DEBUG] Started DeepSeek OCR at {datetime.now().isoformat(timespec='seconds')}")

tokenizer_start_time = time.perf_counter()
tokenizer = AutoTokenizer.from_pretrained(model_name,trust_remote_code=True)
print(f"[DEBUG] Tokenizer loaded in {time.perf_counter() - tokenizer_start_time:.2f}s")

model_start_time = time.perf_counter()
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True,low_cpu_mem_usage=True)
model = model.eval().cuda().to(torch.bfloat16)
print(f"[DEBUG] Model loaded in {time.perf_counter() - model_start_time:.2f}s")

prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_mock_docs")
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "output_ocr", "deepseek_ocr_2")
os.makedirs(output_path, exist_ok=True)
for existing_file in os.listdir(output_path):
    existing_path = os.path.join(output_path, existing_file)
    if os.path.isfile(existing_path):
        os.remove(existing_path)

print(f"[DEBUG] Output directory: {output_path}")

infer_start_time = time.perf_counter()
res = model.infer(tokenizer, prompt=prompt, image_file=image_file, output_path = output_path, base_size = 1024, image_size = 768, crop_mode=True, save_results = False)
combined_output_file = os.path.join(output_path, "deepseek_ocr_2_output.md")
with open(combined_output_file, "w", encoding="utf-8") as f:
    f.write("# DeepSeek-OCR-2 OCR Output\n\n")
    f.write("" if res is None else str(res))

print(f"[DEBUG] Saved combined DeepSeek OCR output to: {combined_output_file}")
print(f"[DEBUG] Inference completed in {time.perf_counter() - infer_start_time:.2f}s")
print(f"[DEBUG] Finished DeepSeek OCR at {datetime.now().isoformat(timespec='seconds')}")
print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s")
