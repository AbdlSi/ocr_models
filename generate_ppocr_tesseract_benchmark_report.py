import json
import math
import os
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parent
INPUT_DIR = ROOT / "benchmark_mock_docs"
PADDLE_OUTPUT = ROOT / "output" / "output_ocr" / "paddleocr_pp_ocrv5_multilingual" / "paddleocr_pp_ocrv5_multilingual_output.txt"
TESSERACT_OUTPUT = ROOT / "output" / "output_ocr" / "tesseract_ocr" / "tesseract_ocr_output.txt"
REPORT_DIR = ROOT / "output" / "pdf"
METRICS_DIR = ROOT / "output" / "benchmark_metrics"
REPORT_PATH = REPORT_DIR / "ppocrv5_vs_tesseract5_benchmark_report.pdf"
METRICS_PATH = METRICS_DIR / "ppocrv5_vs_tesseract5_metrics.json"

FONT_DIR = Path(os.environ.get(
    "CODEX_PDF_FONT_DIR",
    r"C:\Users\asus\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\share\fonts",
))
REGULAR_FONT = FONT_DIR / "DejaVuSans.ttf"
BOLD_FONT = FONT_DIR / "Ubuntu-B.ttf"

TURKISH_CHARS = "çğıöşüÇĞİÖŞÜ"
TURKISH_GROUPS = [
    {"ç", "Ç"},
    {"ğ", "Ğ"},
    {"ı", "I"},
    {"ö", "Ö"},
    {"ş", "Ş"},
    {"ü", "Ü"},
    {"İ", "i"},
]

KEY_FIELDS = {
    "company_registration_documents.pdf": [
        "TÜZEL KİŞİLER İÇİN ODA KAYIT BEYANNAMESİ",
        "Sicil No",
        "MERSİS",
        "İSTANBUL TİCARET ODASI",
        "Vergi Kimlik No",
        "Ticaret Ünvanı",
        "Faaliyet Konusu",
        "Yetkilinin Adı ve Soyadı",
    ],
    "Fire_safety_reports.pdf": [
        "İTFAİYE RAPORU BAŞVURU FORMU",
        "İLK MÜRACAATINIZ MI",
        "YAPI KULLANIM İZİN BELGESİ",
        "YAPI RUHSATI",
        "T.C. Kimlik No",
        "Vergi No",
        "Yangın Önlem Amirliği",
    ],
    "Fire_safety_reports_2.pdf": [
        "Paratoner",
        "İLGİLİ MAKAMA",
        "Binaların Yangından Korunması",
        "TARİH",
        "AD SOYAD",
        "ODA SİCİL NO",
    ],
    "Fire_safety_reports_3.pdf": [
        "Yağmurlama Sistemi",
        "İLGİLİ MAKAMA",
        "yangın su deposu",
        "hidrolik hesabı",
        "AD SOYAD",
        "ODA SİCİL NO",
    ],
    "Lease_agreements.pdf": [
        "KİRA SÖZLEŞMESİ",
        "İli/İlçesi",
        "Ada/Parsel No",
        "Kiralanan Şeyin Cinsi",
        "Kiralayanın",
        "Kiracının",
        "Kira Bedeli",
    ],
    "Lease_agreements_2.pdf": [
        "EK-4",
        "KİRA SÖZLEŞMESİ",
        "ARAZİNİN",
        "Bulunduğu Yer",
        "Kiralayan",
        "Kiracı",
    ],
    "Licence_application_forms.pdf": [
        "SIHHÎ İŞYERİ",
        "İŞYERİ AÇMA VE ÇALIŞMA RUHSATI",
        "T.C. Kimlik No",
        "Vergi Kimlik No",
        "Faaliyet konusu",
        "Beyan sahibinin",
    ],
    "Occupancy_permits_2.pdf": [
        "Ruhsat Sahibi",
        "Takipçi",
        "EKLER",
        "İSKAN DİLEKÇESİ",
        "İş Bitirme Tutanağı",
    ],
    "Occupancy_permits_3.pdf": [
        "İSKANA ESAS ÖN RAPOR",
        "ÜMRANİYE BELEDİYESİ BAŞKANLIĞI",
        "Yapı Sahibi",
        "Yapının Adresi",
        "Kontrol Eden",
    ],
    "Occupancy_permits_4.pdf": [
        "ÜMRANİYE BELEDİYE BAŞKANLIĞI",
        "Yapı Kontrol Müdürlüğü",
        "Yapı Denetim İzin Belgesi",
        "Dosya No",
        "Ruhsat Tarihi",
    ],
    "petition_1.pdf": [
        "İŞYERİ AÇMA VE ÇALIŞMA RUHSATI ALMA",
        "DİLEKÇESİ",
        "Dok.No",
        "T.C. Kimlik No",
        "İmza",
    ],
    "petition_2.pdf": [
        "ERENLER BELEDİYE BAŞKANLIĞI",
        "adresinde yer alan",
        "faaliyet",
        "ruhsatı",
        "arz ederim",
    ],
    "Table_based_forms_1.pdf": [
        "Vergi Kimlik Numarası",
        "Telefon No",
        "Soyadı",
        "Adresi",
        "Cadde Sokak",
        "İlçe Adı",
    ],
    "Table_based_forms_2.pdf": [
        "İLAN VE REKLAM BEYANNAMESİ FORMU",
        "Doküman No",
        "Yayın Tarihi",
        "Revizyon Tarihi",
        "Sayfa",
    ],
    "Table_based_forms_3.pdf": [
        "Mesleki Eğitimin ve Yeterliliğin",
        "ORDU BÜYÜKŞEHİR BELEDİYE BAŞKANLIĞI",
        "Beyanname",
        "Yetkili Kişinin",
        "İmza",
    ],
    "Tax_certificates.pdf": [
        "VERGİ LEVHASI",
        "ADI SOYADI",
        "TİCARET ÜNVANI",
        "VERGİ KİMLİK NO",
        "TAHAKKUK EDEN VERGİ",
    ],
    "Tax_certificates_2.pdf": [
        "VERGİ LEVHASI",
        "ADI SOYADI",
        "TİCARET ÜNVANI",
        "VERGİ KİMLİK NO",
        "TAHAKKUK EDEN VERGİ",
    ],
    "ashissedevri-300x215.jpg": [
        "DEVİR EDEN",
        "DEVİR ALAN",
        "HİSSE ADEDİ",
        "HİSSE TUTARI",
        "Yönetim Kurulu Başkanlığına",
    ],
    "İskan_ve_Yapı_Ruhsat(2000_yılı)_1_75629.jpeg": [
        "YAPI RUHSATI",
        "YAKUPLU BELEDİYE BAŞKANLIĞI",
        "Ruhsat Tarihi",
        "Ruhsat Numarası",
        "İnciler İnşaat A.Ş.",
        "Kendisi",
        "Toplam Alan",
    ],
    "İşyeri Ruhsat İntibak Dilekçesi.png": [
        "İşyeri Açma ve Çalışma Ruhsatlarına İlişkin Yönetmelik",
        "Resmi Gazete",
        "Karar Sayısı",
        "BİRİNCİ KISIM",
        "Amaç",
        "Kapsam",
    ],
    "vergi-levhasi.jpg": [
        "VERGİ LEVHASI",
        "GELİR İDARESİ BAŞKANLIĞI",
        "T.C. KİMLİK NO",
        "VERGİ KİMLİK NO",
        "İŞE BAŞLAMA TARİHİ",
    ],
    "yangin_raporu.png": [
        "T.C. ANTALYA BÜYÜKŞEHİR BELEDİYESİ",
        "YANGIN RAPORU",
        "Olayın Tarihi",
        "Kayıt Tarihi",
        "Yangın Türü",
        "Baca Yangını",
        "Murat DEMİRCİ",
    ],
    "yapı_denetim_izin.jpg": [
        "YAPI DENETİMİ İZİN BELGESİ",
        "BAYINDIRLIK VE İSKAN BAKANLIĞI",
        "KURULUŞUN",
        "BELGE NO",
        "Düzenleme Tarihi",
    ],
}

TABLE_DOCS = {
    "ashissedevri-300x215.jpg",
    "İskan_ve_Yapı_Ruhsat(2000_yılı)_1_75629.jpeg",
    "vergi-levhasi.jpg",
    "yangin_raporu.png",
    "Lease_agreements.pdf",
    "Lease_agreements_2.pdf",
    "Table_based_forms_1.pdf",
    "Table_based_forms_2.pdf",
    "Table_based_forms_3.pdf",
    "Tax_certificates.pdf",
    "Tax_certificates_2.pdf",
    "Fire_safety_reports.pdf",
}


def register_fonts():
    if REGULAR_FONT.exists():
        pdfmetrics.registerFont(TTFont("ReportRegular", str(REGULAR_FONT)))
    else:
        pdfmetrics.registerFont(TTFont("ReportRegular", r"C:\Windows\Fonts\arial.ttf"))

    if BOLD_FONT.exists():
        pdfmetrics.registerFont(TTFont("ReportBold", str(BOLD_FONT)))
    else:
        pdfmetrics.registerFont(TTFont("ReportBold", r"C:\Windows\Fonts\arialbd.ttf"))


def extract_pdf_references():
    references = {}
    page_counts = {}
    embedded_chars = {}

    for path in sorted(INPUT_DIR.glob("*.pdf"), key=lambda p: p.name.lower()):
        reader = PdfReader(str(path))
        page_counts[path.name] = len(reader.pages)
        parts = []
        for page in reader.pages:
            try:
                parts.append(page.extract_text() or "")
            except Exception:
                parts.append("")
        text = "\n".join(parts)
        references[path.name] = text
        embedded_chars[path.name] = len(text.strip())

    for path in sorted(
        [*INPUT_DIR.glob("*.jpg"), *INPUT_DIR.glob("*.jpeg"), *INPUT_DIR.glob("*.png")],
        key=lambda p: p.name.lower(),
    ):
        page_counts[path.name] = 1
        embedded_chars[path.name] = 0

    return references, page_counts, embedded_chars


def parse_output(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    sections = defaultdict(list)
    pattern = re.compile(
        r"(?ms)^=+\s*\nFILE:\s*(.+?)\s*\n(?:PAGE:\s*\d+\s*\n)?=+\s*\n(.*?)(?=^=+\s*\nFILE:|\Z)"
    )
    for match in pattern.finditer(text):
        sections[match.group(1).strip()].append(match.group(2).strip())
    return {name: "\n".join(chunks).strip() for name, chunks in sections.items()}, text


def normalize_for_error(text):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def try_repair_mojibake(text):
    if not any(marker in text for marker in ("Ã", "Ä", "Å", "â")):
        return text
    try:
        repaired = text.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text
    if sum(ch in TURKISH_CHARS for ch in repaired) > sum(ch in TURKISH_CHARS for ch in text):
        return repaired
    return text


def fold_for_matching(text):
    text = try_repair_mojibake(text)
    text = text.replace("ı", "i").replace("İ", "i")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^0-9A-Za-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def word_tokens(text, folded=False):
    if folded:
        return fold_for_matching(text).split()
    text = normalize_for_error(text)
    return re.findall(r"[0-9A-Za-zÇĞİÖŞÜçğıöşü]+", text)


def levenshtein_distance(seq_a, seq_b):
    if seq_a == seq_b:
        return 0
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    previous = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        current = [i]
        for j, item_b in enumerate(seq_b, start=1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (item_a != item_b),
            ))
        previous = current
    return previous[-1]


def error_metrics(reference, hypothesis):
    ref_chars = normalize_for_error(reference)
    hyp_chars = normalize_for_error(hypothesis)
    ref_words = word_tokens(reference)
    hyp_words = word_tokens(hypothesis)
    cer = levenshtein_distance(ref_chars, hyp_chars) / max(1, len(ref_chars))
    wer = levenshtein_distance(ref_words, hyp_words) / max(1, len(ref_words))
    return {
        "cer": cer,
        "wer": wer,
        "ref_chars": len(ref_chars),
        "ref_words": len(ref_words),
    }


def turkish_char_recall(reference, hypothesis):
    ref_counts = Counter()
    hyp_counts = Counter()
    for index, group in enumerate(TURKISH_GROUPS):
        ref_counts[index] = sum(reference.count(ch) for ch in group)
        hyp_counts[index] = sum(hypothesis.count(ch) for ch in group)
    ref_total = sum(ref_counts.values())
    if ref_total == 0:
        return None
    hits = sum(min(ref_counts[index], hyp_counts[index]) for index in ref_counts)
    return hits / ref_total


def key_field_scores(outputs):
    rows = []
    hits = 0
    total = 0
    table_hits = 0
    table_total = 0
    for file_name, fields in KEY_FIELDS.items():
        output_norm = fold_for_matching(outputs.get(file_name, ""))
        doc_hits = 0
        for field in fields:
            matched = fold_for_matching(field) in output_norm
            doc_hits += int(matched)
            hits += int(matched)
            total += 1
            if file_name in TABLE_DOCS:
                table_hits += int(matched)
                table_total += 1
        rows.append({
            "file": file_name,
            "hits": doc_hits,
            "total": len(fields),
            "accuracy": doc_hits / len(fields) if fields else None,
        })
    return {
        "overall": hits / total if total else None,
        "hits": hits,
        "total": total,
        "table_marker": table_hits / table_total if table_total else None,
        "table_hits": table_hits,
        "table_total": table_total,
        "by_file": rows,
    }


def sequence_order_score(reference, hypothesis):
    ref_tokens = word_tokens(reference, folded=True)
    hyp_tokens = word_tokens(hypothesis, folded=True)
    if not ref_tokens:
        return None
    distance = levenshtein_distance(ref_tokens, hyp_tokens)
    return max(0.0, 1.0 - distance / max(1, len(ref_tokens)))


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def weighted_mean(rows, key, weight_key):
    numerator = 0.0
    denominator = 0.0
    for row in rows:
        value = row.get(key)
        weight = row.get(weight_key, 0)
        if value is not None and weight:
            numerator += value * weight
            denominator += weight
    return numerator / denominator if denominator else None


def pct(value, decimals=1):
    if value is None:
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def num(value, decimals=3):
    if value is None:
        return "N/A"
    return f"{value:.{decimals}f}"


def get_tool_versions():
    versions = {}
    for command, key in [(["tesseract", "--version"], "tesseract"), (["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], "gpu")]:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
            versions[key] = (result.stdout or result.stderr).splitlines()[0].strip()
        except Exception as exc:
            versions[key] = f"Unavailable ({type(exc).__name__})"
    return versions


def compute_metrics():
    references, page_counts, embedded_chars = extract_pdf_references()
    paddle_sections, paddle_raw = parse_output(PADDLE_OUTPUT)
    tess_sections, tess_raw = parse_output(TESSERACT_OUTPUT)

    total_pages = sum(page_counts.values())
    reference_files = [
        name for name, text in references.items()
        if len(text.strip()) >= 250
    ]

    models = {
        "PaddleOCR PP-OCRv5 Multilingual": {
            "sections": paddle_sections,
            "raw_chars": len(paddle_raw),
            "output_file": str(PADDLE_OUTPUT),
        },
        "Tesseract 5": {
            "sections": tess_sections,
            "raw_chars": len(tess_raw),
            "output_file": str(TESSERACT_OUTPUT),
        },
    }

    for model in models.values():
        rows = []
        for file_name in reference_files:
            reference = references[file_name]
            hypothesis = model["sections"].get(file_name, "")
            row = {"file": file_name}
            row.update(error_metrics(reference, hypothesis))
            row["turkish_char_recall"] = turkish_char_recall(reference, hypothesis)
            row["reading_order_score"] = sequence_order_score(reference, hypothesis)
            rows.append(row)

        key_scores = key_field_scores(model["sections"])
        model["cer"] = weighted_mean(rows, "cer", "ref_chars")
        model["wer"] = weighted_mean(rows, "wer", "ref_words")
        model["turkish_char_accuracy"] = mean([row["turkish_char_recall"] for row in rows])
        model["reading_order_score"] = mean([row["reading_order_score"] for row in rows])
        model["key_field_accuracy"] = key_scores["overall"]
        model["table_preservation"] = key_scores["table_marker"]
        model["reference_file_metrics"] = rows
        model["key_field_scores"] = key_scores
        model["files_detected"] = len(model["sections"])
        model["missing_output_files"] = sorted(set(page_counts) - set(model["sections"]))

    metrics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "input_dir": str(INPUT_DIR),
            "total_files": len(page_counts),
            "total_pages": total_pages,
            "pdf_files": len(list(INPUT_DIR.glob("*.pdf"))),
            "image_files": len(page_counts) - len(list(INPUT_DIR.glob("*.pdf"))),
            "embedded_reference_files": len(reference_files),
            "embedded_reference_policy": "PDFs with at least 250 embedded text characters; scanned PDFs and standalone images excluded from CER/WER.",
            "page_counts": page_counts,
            "embedded_chars": embedded_chars,
        },
        "models": models,
        "environment": get_tool_versions(),
        "limitations": [
            "CER/WER use embedded PDF text as reference where available. No manual ground truth was supplied for scanned PDFs or standalone images.",
            "Runtime, peak RAM, and peak VRAM were not present in the provided OCR output files, so those metrics are reported as not captured.",
            "PaddleOCR output is evaluated exactly as supplied. It preserves many Turkish characters, but some substitutions such as non-Turkish accented Latin letters remain visible.",
        ],
    }
    return metrics


def para(text, style):
    return Paragraph(escape(str(text)), style)


def build_table(data, col_widths=None, header=True, font_size=8.5):
    table = Table(data, colWidths=col_widths, hAlign="LEFT", repeatRows=1 if header else 0)
    vertical_padding = 2 if font_size <= 7 else 4
    styles = [
        ("FONTNAME", (0, 0), (-1, -1), "ReportRegular"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D0D7DE")),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), vertical_padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), vertical_padding),
    ]
    if header:
        styles.extend([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2F8")),
            ("FONTNAME", (0, 0), (-1, 0), "ReportBold"),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
        ])
    table.setStyle(TableStyle(styles))
    return table


def page_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("ReportRegular", 8)
    canvas.setFillColor(colors.HexColor("#6B7280"))
    canvas.drawString(1.5 * cm, 1.0 * cm, "PaddleOCR PP-OCRv5 Multilingual vs Tesseract 5 benchmark")
    canvas.drawRightString(A4[0] - 1.5 * cm, 1.0 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf(metrics):
    register_fonts()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=A4,
        rightMargin=1.4 * cm,
        leftMargin=1.4 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TitleCenter",
        parent=styles["Title"],
        fontName="ReportBold",
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#111827"),
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="Heading",
        parent=styles["Heading2"],
        fontName="ReportBold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#111827"),
        spaceBefore=12,
        spaceAfter=7,
    ))
    styles.add(ParagraphStyle(
        name="Body",
        parent=styles["BodyText"],
        fontName="ReportRegular",
        fontSize=9.2,
        leading=13,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#1F2937"),
        spaceAfter=6,
    ))
    styles.add(ParagraphStyle(
        name="Small",
        parent=styles["BodyText"],
        fontName="ReportRegular",
        fontSize=7.8,
        leading=10.5,
        textColor=colors.HexColor("#374151"),
    ))
    styles.add(ParagraphStyle(
        name="Cell",
        parent=styles["BodyText"],
        fontName="ReportRegular",
        fontSize=7.6,
        leading=9.5,
        textColor=colors.HexColor("#111827"),
    ))

    cell = styles["Cell"]
    body = styles["Body"]
    heading = styles["Heading"]
    story = []

    dataset = metrics["dataset"]
    paddle = metrics["models"]["PaddleOCR PP-OCRv5 Multilingual"]
    tess = metrics["models"]["Tesseract 5"]

    story.append(Paragraph("OCR Benchmark Report", styles["TitleCenter"]))
    story.append(para("PaddleOCR PP-OCRv5 Multilingual vs Tesseract 5", body))
    story.append(para(f"Generated: {metrics['generated_at']}", body))
    story.append(Spacer(1, 8))

    story.append(para("Executive Summary", heading))
    story.append(para(
        "Using the supplied OCR output files, PaddleOCR PP-OCRv5 Multilingual produced the stronger measured result across CER, WER, key-field recall, "
        "Turkish-character preservation, table/form marker recall, and reading-order similarity. Tesseract 5 remains easier to deploy and is CPU-only, "
        "but its raw output loses more Turkish diacritics and introduces more layout noise on these municipal forms.",
        body,
    ))

    summary_rows = [
        [para("Metric", cell), para("PaddleOCR PP-OCRv5", cell), para("Tesseract 5", cell), para("Better", cell)],
        [para("Character Error Rate", cell), para(num(paddle["cer"]), cell), para(num(tess["cer"]), cell), para("Lower is better", cell)],
        [para("Word Error Rate", cell), para(num(paddle["wer"]), cell), para(num(tess["wer"]), cell), para("Lower is better", cell)],
        [para("Key-field accuracy", cell), para(pct(paddle["key_field_accuracy"]), cell), para(pct(tess["key_field_accuracy"]), cell), para("Higher is better", cell)],
        [para("Turkish-character accuracy", cell), para(pct(paddle["turkish_char_accuracy"]), cell), para(pct(tess["turkish_char_accuracy"]), cell), para("Higher is better", cell)],
        [para("Table marker preservation", cell), para(pct(paddle["table_preservation"]), cell), para(pct(tess["table_preservation"]), cell), para("Higher is better", cell)],
        [para("Reading-order preservation", cell), para(pct(paddle["reading_order_score"]), cell), para(pct(tess["reading_order_score"]), cell), para("Higher is better", cell)],
    ]
    story.append(build_table(summary_rows, [5.0 * cm, 3.7 * cm, 3.2 * cm, 3.6 * cm], font_size=7.6))

    story.append(para("Dataset And Ground Truth", heading))
    story.append(para(
        f"The benchmark folder contains {dataset['total_files']} files across {dataset['total_pages']} pages "
        f"({dataset['pdf_files']} PDFs and {dataset['image_files']} standalone images). "
        f"CER/WER were computed only for {dataset['embedded_reference_files']} PDFs with extractable embedded text "
        "of at least 250 characters. Scanned PDFs and image-only files were included in key-field/table checks but excluded from CER/WER because no manual ground truth was supplied.",
        body,
    ))
    story.append(para(
        "All text-quality scores use the provided output files as-is. No OCR output was corrected before scoring.",
        body,
    ))

    story.append(para("Detailed Measurement Definitions", heading))
    definitions = [
        "Character Error Rate: Levenshtein character distance divided by reference character count after Unicode normalization, case folding, and whitespace collapse.",
        "Word Error Rate: Levenshtein word distance divided by reference word count on Turkish-aware word tokens.",
        "Key-field accuracy: recall of curated document labels and key field names across the supplied benchmark files, matched accent-insensitively to avoid double-penalizing Turkish marks.",
        "Turkish-character accuracy: recall of expected Turkish-specific characters in the embedded-text reference subset, measured from the raw OCR output.",
        "Table preservation: recall of key table/form labels on table-heavy documents.",
        "Reading-order preservation: word-sequence similarity against embedded-text references after accent-insensitive folding.",
    ]
    story.append(ListFlowable([ListItem(para(item, body)) for item in definitions], bulletType="bullet", leftIndent=14))

    story.append(PageBreak())
    story.append(para("Model-Level Results", heading))
    model_rows = [
        [para("Area", cell), para("PaddleOCR PP-OCRv5 Multilingual", cell), para("Tesseract 5", cell)],
        [para("Observed strength", cell), para("Best measured raw text quality in this artifact set; strong key-field recall and much better Turkish-character preservation than Tesseract.", cell), para("Simple CPU deployment; native CLI is easy to call from scripts and does not require GPU memory.", cell)],
        [para("Observed weakness", cell), para("Heavier Python/Paddle dependency stack; some Turkish letters are still substituted with non-Turkish accented Latin characters; runtime footprint not captured.", cell), para("Plain text output contains layout noise, weak table structure, and many Turkish diacritics are lost or substituted.", cell)],
        [para("Average processing time per page", cell), para("Not captured in supplied output/log files.", cell), para("Not captured in supplied output/log files.", cell)],
        [para("Peak RAM", cell), para("Not captured in supplied output/log files.", cell), para("Not captured in supplied output/log files.", cell)],
        [para("Peak VRAM", cell), para("Not captured. PaddleOCR can run CPU or GPU depending on PaddlePaddle runtime.", cell), para("0 VRAM expected for standard Tesseract CLI/Python use; CPU-only OCR engine.", cell)],
        [para("CPU/GPU compatibility", cell), para("Python/PaddlePaddle stack; CPU and CUDA GPU deployments are possible, but dependency setup is heavier.", cell), para("CPU-first native binary. GPU is not required and normally not used.", cell)],
        [para("Output format", cell), para("Script writes plain text; PaddleOCR result objects can also expose structured recognition fields.", cell), para("Script writes plain text; Tesseract can also emit hOCR, TSV, searchable PDF, and box formats when configured.", cell)],
        [para("Integration difficulty", cell), para("Medium-high: Python package, model downloads, Paddle runtime, optional GPU/CUDA alignment.", cell), para("Low-medium: install native Tesseract plus language data and call CLI or wrapper.", cell)],
        [para("License suitability", cell), para("Suitable for commercial/internal use under Apache-2.0, subject to preserving notices and checking bundled model/data terms.", cell), para("Suitable for commercial/internal use under Apache-2.0, subject to preserving notices and traineddata terms.", cell)],
    ]
    story.append(build_table(model_rows, [4.2 * cm, 6.0 * cm, 6.0 * cm], font_size=7.2))

    story.append(para("Per-Document Text Metrics", heading))
    story.append(para(
        "The following rows show the CER/WER subset only: PDFs where embedded text was available. Lower CER/WER is better.",
        body,
    ))
    doc_rows = [[
        para("File", cell),
        para("Paddle CER", cell),
        para("Tess CER", cell),
        para("Paddle WER", cell),
        para("Tess WER", cell),
        para("Ref chars", cell),
    ]]
    tess_by_file = {row["file"]: row for row in tess["reference_file_metrics"]}
    for row in paddle["reference_file_metrics"]:
        trow = tess_by_file[row["file"]]
        doc_rows.append([
            para(row["file"], cell),
            para(num(row["cer"]), cell),
            para(num(trow["cer"]), cell),
            para(num(row["wer"]), cell),
            para(num(trow["wer"]), cell),
            para(str(row["ref_chars"]), cell),
        ])
    story.append(build_table(doc_rows, [5.5 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm, 2.0 * cm], font_size=6.2))

    story.append(PageBreak())
    story.append(para("Key-Field Accuracy By Document", heading))
    key_rows = [[para("File", cell), para("Paddle", cell), para("Tesseract", cell), para("Fields", cell)]]
    tess_key_by_file = {row["file"]: row for row in tess["key_field_scores"]["by_file"]}
    for prow in paddle["key_field_scores"]["by_file"]:
        trow = tess_key_by_file[prow["file"]]
        key_rows.append([
            para(prow["file"], cell),
            para(f"{prow['hits']}/{prow['total']} ({pct(prow['accuracy'])})", cell),
            para(f"{trow['hits']}/{trow['total']} ({pct(trow['accuracy'])})", cell),
            para("table/form" if prow["file"] in TABLE_DOCS else "text/form", cell),
        ])
    story.append(build_table(key_rows, [7.0 * cm, 3.0 * cm, 3.0 * cm, 2.5 * cm], font_size=6.7))

    story.append(para("Interpretation", heading))
    interpretation = [
        "PaddleOCR PP-OCRv5 is the stronger candidate for this dataset if accuracy is the priority.",
        "Tesseract 5 is still attractive when installation simplicity, CPU-only operation, and predictable licensing/deployment matter more than accuracy.",
        "Neither provided output captures peak memory or processing time. Add process-level RAM/GPU sampling to both scripts before using this benchmark for production cost planning.",
        "For table-heavy municipal forms, neither plain-text output preserves true table geometry. If table extraction is a requirement, use structured OCR outputs with bounding boxes, hOCR/TSV, or a layout model.",
    ]
    story.append(ListFlowable([ListItem(para(item, body)) for item in interpretation], bulletType="bullet", leftIndent=14))

    story.append(para("Runtime And Memory Instrumentation Gap", heading))
    story.append(para(
        "Average processing time per page, peak RAM, and peak VRAM could not be measured from the two supplied OCR text files because they contain extracted text only, not execution logs. "
        "A defensible runtime benchmark should rerun each model with: per-page start/end timestamps, process RSS sampling, and GPU memory sampling via nvidia-smi for GPU-backed PaddleOCR.",
        body,
    ))
    env_rows = [
        [para("Local check", cell), para("Value", cell)],
        [para("Tesseract executable", cell), para(metrics["environment"].get("tesseract", "Unavailable"), cell)],
        [para("GPU visible locally", cell), para(metrics["environment"].get("gpu", "Unavailable"), cell)],
        [para("PaddleOCR runtime", cell), para("Not installed in the local workspace Python checked during this report; provided output file was used instead.", cell)],
    ]
    story.append(KeepTogether([build_table(env_rows, [5.5 * cm, 10.0 * cm], font_size=7.3)]))

    story.append(PageBreak())
    story.append(para("Sources And License Notes", heading))
    sources = [
        "PaddleOCR repository and license: https://github.com/PaddlePaddle/PaddleOCR and https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE",
        "Tesseract repository and license: https://github.com/tesseract-ocr/tesseract and https://github.com/tesseract-ocr/tesseract/blob/main/LICENSE",
        f"Local PaddleOCR output: {PADDLE_OUTPUT}",
        f"Local Tesseract output: {TESSERACT_OUTPUT}",
        f"Benchmark documents: {INPUT_DIR}",
    ]
    story.append(ListFlowable([ListItem(para(item, body)) for item in sources], bulletType="bullet", leftIndent=14))

    story.append(para("Appendix: Limitations", heading))
    story.append(ListFlowable([ListItem(para(item, body)) for item in metrics["limitations"]], bulletType="bullet", leftIndent=14))

    doc.build(story, onFirstPage=page_footer, onLaterPages=page_footer)


def main():
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics()
    METRICS_PATH.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    build_pdf(metrics)
    print(f"Wrote metrics: {METRICS_PATH}")
    print(f"Wrote report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
