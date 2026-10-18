"""
extractor.py - Image Preparation and Model Extraction with Retry and Caching
Uses local open-weight vision models via Ollama. 100% offline, zero cloud calls.
"""

import hashlib
import io
import json
import math
import re
import time
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image, ImageOps
import ollama

DEFAULT_MODEL_NAME = "gemma3:4b"

DEFAULT_CATEGORIES = [
    "Software",
    "Travel",
    "Meals",
    "Equipment",
    "Office",
    "Utilities",
    "Other",
]

ALLOWED_CONFIDENCE = ["low", "medium", "high"]

PRIMARY_PROMPT = (
    "You are a receipt reader. Look at this receipt or invoice and return ONLY "
    "a JSON object with these keys: vendor (string), date (YYYY-MM-DD, or empty "
    "string if unclear), total_amount (number, no currency symbol), tax_amount "
    "(number, no currency symbol, or 0), currency (3-letter code like INR, USD, "
    "EUR, or empty), category (exactly one of: Software, Travel, Meals, Equipment, "
    "Office, Utilities, Other), line_items_summary (short string describing items, "
    "max 80 chars), confidence (low, medium, or high). Do not add any text before "
    "or after the JSON. If a field is unreadable, use an empty string or 0."
)

STRICT_RETRY_PROMPT = (
    "CRITICAL: Your previous response was not valid JSON. Return ONLY a valid JSON "
    "object, with NO explanation, NO markdown formatting, and NO text outside the "
    "curly braces {}.\n"
    "Required keys: vendor (string), date (YYYY-MM-DD), total_amount (number), "
    "tax_amount (number), currency (3-letter code), category (one of: Software, "
    "Travel, Meals, Equipment, Office, Utilities, Other), line_items_summary "
    "(short string max 80 chars), confidence (low, medium, or high)."
)


def compute_image_hash(image_bytes: bytes) -> str:
    """Computes a SHA-256 hash of the image bytes for session-level caching."""
    return hashlib.sha256(image_bytes).hexdigest()


def prepare_image(image_file: Any) -> Tuple[bytes, Image.Image]:
    """
    Opens an image with Pillow, fixes orientation via EXIF, converts to RGB,
    resizes longest side to at most 1280px maintaining aspect ratio, and returns
    (compressed_jpeg_bytes, pil_thumbnail_image).
    """
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image)

    if image.mode != "RGB":
        image = image.convert("RGB")

    max_side = 1280
    width, height = image.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (int(round(width * scale)), int(round(height * scale)))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue(), image


def _call_ollama_chat(
    image_bytes: bytes,
    prompt: str,
    model_name: str,
) -> str:
    """Invokes ollama.chat with zero temperature."""
    response = ollama.chat(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_bytes],
            }
        ],
        options={"temperature": 0},
    )

    if hasattr(response, "message") and hasattr(response.message, "content"):
        return response.message.content or ""
    elif isinstance(response, dict):
        return response.get("message", {}).get("content", "")
    return str(response)


def parse_receipt_response(
    raw_text: str,
    filename: str = "",
    allowed_categories: Optional[List[str]] = None,
) -> Tuple[Optional[Dict[str, Any]], bool]:
    """
    Attempts to safely parse model text into a validated receipt dict.
    Returns (result_dict, is_valid_json).
    If parsing fails completely, returns (fallback_dict, False).
    """
    categories = allowed_categories or DEFAULT_CATEGORIES
    fallback_row = {
        "file_name": filename,
        "vendor": "NEEDS REVIEW",
        "date": "",
        "total_amount": 0.0,
        "tax_amount": 0.0,
        "currency": "",
        "category": "Other",
        "client_project": "",
        "tax_deductible": False,
        "line_items_summary": "",
        "confidence": "low",
        "notes": raw_text.strip()[:500] if raw_text else "Empty response from model",
    }

    if not raw_text or not isinstance(raw_text, str):
        return fallback_row, False

    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace < first_brace:
        return fallback_row, False

    json_str = cleaned[first_brace : last_brace + 1]

    try:
        data = json.loads(json_str)
        if not isinstance(data, dict):
            return fallback_row, False
    except Exception:
        return fallback_row, False

    # Vendor
    vendor_val = data.get("vendor")
    vendor = str(vendor_val).strip() if vendor_val is not None else ""
    if not vendor:
        vendor = "NEEDS REVIEW"

    # Date
    date_val = data.get("date")
    date_str = str(date_val).strip() if date_val is not None else ""
    if date_str.lower() in ["none", "null", "unclear", "n/a"]:
        date_str = ""

    # Total amount
    raw_amount = data.get("total_amount", 0.0)
    try:
        if isinstance(raw_amount, str):
            clean_num = re.sub(r"[^\d.-]", "", raw_amount)
            total_amount = float(clean_num) if clean_num else 0.0
        else:
            total_amount = float(raw_amount)
        if math.isnan(total_amount) or math.isinf(total_amount):
            total_amount = 0.0
    except (ValueError, TypeError):
        total_amount = 0.0

    # Tax amount
    raw_tax = data.get("tax_amount", 0.0)
    try:
        if isinstance(raw_tax, str):
            clean_tax = re.sub(r"[^\d.-]", "", raw_tax)
            tax_amount = float(clean_tax) if clean_tax else 0.0
        else:
            tax_amount = float(raw_tax)
        if math.isnan(tax_amount) or math.isinf(tax_amount):
            tax_amount = 0.0
    except (ValueError, TypeError):
        tax_amount = 0.0

    # Currency
    currency_val = data.get("currency")
    currency = str(currency_val).strip().upper() if currency_val is not None else ""
    if currency.lower() in ["none", "null", "n/a"] or len(currency) > 4:
        currency = currency[:3] if len(currency) >= 3 else ""

    # Category validation
    category_val = str(data.get("category", "")).strip()
    category = "Other"
    for cat in categories:
        if category_val.lower() == cat.lower():
            category = cat
            break

    # Confidence validation
    confidence_val = str(data.get("confidence", "")).strip().lower()
    confidence = confidence_val if confidence_val in ALLOWED_CONFIDENCE else "low"

    # Line items summary (max 80 chars)
    line_items = str(data.get("line_items_summary", "")).strip()[:80]

    # Notes
    notes_val = data.get("notes")
    notes = str(notes_val).strip() if notes_val is not None else ""

    return {
        "file_name": filename,
        "vendor": vendor,
        "date": date_str,
        "total_amount": round(total_amount, 2),
        "tax_amount": round(tax_amount, 2),
        "currency": currency,
        "category": category,
        "client_project": str(data.get("client_project", "")).strip(),
        "tax_deductible": False,
        "line_items_summary": line_items,
        "confidence": confidence,
        "notes": notes,
    }, True


def extract_receipt_with_retry(
    image_bytes: bytes,
    filename: str = "",
    model_name: str = DEFAULT_MODEL_NAME,
    cache: Optional[Dict[str, Dict[str, Any]]] = None,
    allowed_categories: Optional[List[str]] = None,
) -> Tuple[Dict[str, Any], bool, float]:
    """
    Extracts receipt details with caching and automatic retry:
    1. Checks if image SHA-256 hash exists in cache.
    2. Makes initial call with PRIMARY_PROMPT.
    3. If invalid JSON, retries once with STRICT_RETRY_PROMPT.
    4. If still invalid, returns safe fallback marked NEEDS REVIEW.
    Returns (receipt_data_dict, was_cached, elapsed_seconds).
    """
    img_hash = compute_image_hash(image_bytes)

    # Step 4: Cache lookup
    if cache is not None and img_hash in cache:
        cached_result = dict(cache[img_hash])
        cached_result["file_name"] = filename
        return cached_result, True, 0.0

    start_time = time.time()
    raw_reply = ""
    try:
        raw_reply = _call_ollama_chat(image_bytes, PRIMARY_PROMPT, model_name)
    except Exception as e:
        elapsed = time.time() - start_time
        fallback, _ = parse_receipt_response(f"Model call failed: {e}", filename=filename)
        fallback["image_hash"] = img_hash
        return fallback, False, elapsed

    parsed, is_valid = parse_receipt_response(
        raw_reply, filename=filename, allowed_categories=allowed_categories
    )

    # Step 4: Retry once with stricter prompt if first parse failed
    if not is_valid:
        try:
            retry_reply = _call_ollama_chat(image_bytes, STRICT_RETRY_PROMPT, model_name)
            retry_parsed, retry_valid = parse_receipt_response(
                retry_reply, filename=filename, allowed_categories=allowed_categories
            )
            if retry_valid:
                parsed = retry_parsed
                parsed["notes"] = (parsed.get("notes", "") + " (Extracted via retry)").strip()
        except Exception:
            pass

    elapsed = time.time() - start_time
    parsed["image_hash"] = img_hash

    # Save to session cache
    if cache is not None:
        cache[img_hash] = dict(parsed)

    return parsed, False, elapsed


def test_model_connection(model_name: str) -> Tuple[bool, float, str]:
    """
    Tests model reachability and response latency with a lightweight ping.
    Returns (success: bool, latency_ms: float, message: str).
    """
    start = time.time()
    try:
        resp = ollama.chat(
            model=model_name,
            messages=[{"role": "user", "content": "Respond with the single word 'OK'."}],
            options={"temperature": 0},
        )
        latency = (time.time() - start) * 1000.0
        content = ""
        if hasattr(resp, "message") and hasattr(resp.message, "content"):
            content = resp.message.content or ""
        elif isinstance(resp, dict):
            content = resp.get("message", {}).get("content", "")
        return True, latency, f"Model responded in {latency:.0f} ms: '{content.strip()}'"
    except Exception as e:
        latency = (time.time() - start) * 1000.0
        return False, latency, str(e)


def check_ollama_status(target_model: str) -> Tuple[bool, bool, List[str]]:
    """
    Checks if Ollama is running and whether target_model is installed.
    Returns (server_running, model_installed, list_of_models).
    """
    try:
        res = ollama.list()
        models = []
        if hasattr(res, "models"):
            models = [getattr(m, "model", str(m)) for m in res.models]
        elif isinstance(res, dict):
            models = [m.get("model") or m.get("name", "") for m in res.get("models", [])]

        target = target_model.strip().lower()
        model_found = False
        for m in models:
            m_lower = m.lower()
            if target == m_lower:
                model_found = True
                break
            if ":" not in target and m_lower.startswith(f"{target}:"):
                model_found = True
                break
            if ":" in target and m_lower == target:
                model_found = True
                break

        return True, model_found, models
    except Exception:
        return False, False, []
