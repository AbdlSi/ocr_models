import gc
import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:64")

import torch
from pdf2image import convert_from_path, pdfinfo_from_path
from PIL import Image, ImageOps
from transformers import AutoModel, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "benchmark_mock_docs"
OUTPUT_DIR = BASE_DIR / "output" / "output_ocr" / "deepseek_ocr_2"
OUTPUT_FILE = OUTPUT_DIR / "deepseek_ocr_2_output.md"

MODEL_NAME = "deepseek-ai/DeepSeek-OCR-2"
PROMPT = "<image>\n<|grounding|>Convert the document to markdown. "
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}

# Low-VRAM defaults for Colab T4-class GPUs. Increase these only if OCR quality is too low.
LOW_VRAM = os.getenv("DEEPSEEK_LOW_VRAM", "1") != "0"
if LOW_VRAM:
    PDF_DPI = 150
    BASE_SIZE = 768
    IMAGE_SIZE = 512
    MAX_IMAGE_SIDE = 1600
    CROP_MODE = False
else:
    PDF_DPI = 200
    BASE_SIZE = 1024
    IMAGE_SIZE = 768
    MAX_IMAGE_SIDE = 2200
    CROP_MODE = True

SAVE_RESULTS = False
RETRY_BASE_SIZE = 512
RETRY_IMAGE_SIZE = 384

IMAGE_MARKDOWN_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_IMAGE_RE = re.compile(r"<img\b[^>]*>", flags=re.IGNORECASE)
RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def clean_output_dir(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for existing_path in output_dir.iterdir():
        if existing_path.is_file() or existing_path.is_symlink():
            existing_path.unlink()
        elif existing_path.is_dir():
            shutil.rmtree(existing_path)


def strip_image_references(text):
    text = IMAGE_MARKDOWN_RE.sub("", text)
    text = HTML_IMAGE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_text_artifacts(root_dir):
    chunks = []
    for path in sorted(Path(root_dir).rglob("*")):
        if path.is_file() and path.suffix.lower() in {".md", ".markdown", ".txt"}:
            chunks.append(strip_image_references(path.read_text(encoding="utf-8", errors="ignore")))
    return "\n\n".join(chunk for chunk in chunks if chunk)


def safe_stem(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._") or "document"


def file_header(path):
    return f"\n\n====================\nFILE: {path.relative_to(INPUT_DIR).as_posix()}\n====================\n"


def page_header(page_number):
    return f"\n--- Page {page_number} ---\n"


def append_output(text):
    with OUTPUT_FILE.open("a", encoding="utf-8") as output_stream:
        output_stream.write(text)


def log_cuda_memory(label):
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"[DEBUG] CUDA memory {label}: allocated={allocated:.2f} GiB, reserved={reserved:.2f} GiB", flush=True)


def save_resized_rgb(image, image_path):
    image = ImageOps.exif_transpose(image).convert("RGB")
    max_side = max(image.size)
    if max_side > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max_side
        new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
        image = image.resize(new_size, RESAMPLE_LANCZOS)
    image.save(image_path, optimize=True)


def iter_image_inputs(input_path, work_dir):
    input_path = Path(input_path)
    work_dir = Path(work_dir)

    if input_path.suffix.lower() == ".pdf":
        page_count = int(pdfinfo_from_path(str(input_path)).get("Pages", 0))
        for page_number in range(1, page_count + 1):
            page_image = convert_from_path(
                str(input_path),
                dpi=PDF_DPI,
                first_page=page_number,
                last_page=page_number,
            )[0]
            image_path = work_dir / f"{safe_stem(input_path)}_page_{page_number:04d}.png"
            save_resized_rgb(page_image, image_path)
            page_image.close()
            yield page_number, image_path
    else:
        image_path = work_dir / f"{safe_stem(input_path)}_image.png"
        with Image.open(input_path) as image:
            save_resized_rgb(image, image_path)
        yield None, image_path


def run_deepseek_image(model, tokenizer, dtype, image_path, base_size, image_size, crop_mode):
    with tempfile.TemporaryDirectory() as staging_dir:
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
            result = model.infer(
                tokenizer,
                prompt=PROMPT,
                image_file=str(image_path),
                output_path=staging_dir,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=SAVE_RESULTS,
            )

        if result is not None and str(result).strip():
            return strip_image_references(str(result))

        text = read_text_artifacts(staging_dir)
        return text if text else "(No text returned.)"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("DeepSeek-OCR-2 needs a CUDA GPU runtime.")

    clean_output_dir(OUTPUT_DIR)

    input_files = sorted(
        [path for path in INPUT_DIR.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS],
        key=lambda path: path.relative_to(INPUT_DIR).as_posix().lower(),
    )
    if not input_files:
        raise FileNotFoundError(f"No supported PDF/image files found under: {INPUT_DIR}")

    start_time = time.perf_counter()
    print(f"[DEBUG] Started DeepSeek OCR at {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"[DEBUG] Output directory: {OUTPUT_DIR}", flush=True)
    print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        "[DEBUG] Low-VRAM settings: "
        f"dpi={PDF_DPI}, base_size={BASE_SIZE}, image_size={IMAGE_SIZE}, "
        f"max_image_side={MAX_IMAGE_SIDE}, crop_mode={CROP_MODE}, save_results={SAVE_RESULTS}",
        flush=True,
    )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[DEBUG] DeepSeek dtype: {dtype}", flush=True)

    tokenizer_start_time = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print(f"[DEBUG] Tokenizer loaded in {time.perf_counter() - tokenizer_start_time:.2f}s", flush=True)

    model_start_time = time.perf_counter()
    try:
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            _attn_implementation="flash_attention_2",
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )
    except Exception as exc:
        print(f"[DEBUG] Flash attention load failed; falling back. Reason: {exc}", flush=True)
        model = AutoModel.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            torch_dtype=dtype,
        )

    model = model.eval().requires_grad_(False).cuda().to(dtype)
    torch.cuda.empty_cache()
    log_cuda_memory("after model load")
    print(f"[DEBUG] Model loaded in {time.perf_counter() - model_start_time:.2f}s", flush=True)

    OUTPUT_FILE.write_text("# DeepSeek-OCR-2 OCR Output\n", encoding="utf-8")

    infer_start_time = time.perf_counter()
    for file_index, input_path in enumerate(input_files, start=1):
        file_start_time = time.perf_counter()
        print(f"[DEBUG] Processing {file_index}/{len(input_files)}: {input_path.name}", flush=True)
        append_output(file_header(input_path))

        try:
            with tempfile.TemporaryDirectory() as conversion_dir:
                for page_number, image_path in iter_image_inputs(input_path, conversion_dir):
                    page_start_time = time.perf_counter()
                    page_label = f" page {page_number}" if page_number is not None else ""
                    if page_number is not None:
                        append_output(page_header(page_number))

                    try:
                        text = run_deepseek_image(
                            model,
                            tokenizer,
                            dtype,
                            image_path,
                            base_size=BASE_SIZE,
                            image_size=IMAGE_SIZE,
                            crop_mode=CROP_MODE,
                        )
                    except torch.cuda.OutOfMemoryError:
                        print(f"[DEBUG] CUDA OOM on {input_path.name}{page_label}; retrying smaller", flush=True)
                        gc.collect()
                        torch.cuda.empty_cache()
                        text = run_deepseek_image(
                            model,
                            tokenizer,
                            dtype,
                            image_path,
                            base_size=RETRY_BASE_SIZE,
                            image_size=RETRY_IMAGE_SIZE,
                            crop_mode=False,
                        )

                    append_output(text.strip() + "\n")
                    print(
                        f"[DEBUG] Finished {input_path.name}{page_label} in "
                        f"{time.perf_counter() - page_start_time:.2f}s",
                        flush=True,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    log_cuda_memory(f"after {input_path.name}{page_label}")

            print(f"[DEBUG] File completed in {time.perf_counter() - file_start_time:.2f}s: {input_path.name}", flush=True)
        except Exception as exc:
            error_text = f"[ERROR] {type(exc).__name__}: {exc}"
            append_output(error_text + "\n")
            print(f"[DEBUG] Failed for {input_path.name}: {error_text}", flush=True)

    print(f"[DEBUG] Saved DeepSeek OCR output to: {OUTPUT_FILE}", flush=True)
    print(f"[DEBUG] Inference completed in {time.perf_counter() - infer_start_time:.2f}s", flush=True)
    print(f"[DEBUG] Finished DeepSeek OCR at {datetime.now().isoformat(timespec='seconds')}", flush=True)
    print(f"[DEBUG] Total elapsed time: {time.perf_counter() - start_time:.2f}s", flush=True)


if __name__ == "__main__":
    main()
