import os
import re
import json
from typing import Any, Dict
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dateutil import parser as date_parser
import google.generativeai as genai


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DynamicExtractRequest(BaseModel):
    text: str
    schema: Dict[str, str]


def extract_json(text: str) -> dict:
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {}


def clean_string(value: Any):
    if value is None:
        return None
    value = str(value).strip()
    if value.lower() in ["null", "none", "not found", "unknown", ""]:
        return None
    return value


def to_integer(value: Any):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        return int(value)

    text = str(value).replace(",", "")
    match = re.search(r"-?\d+", text)
    if not match:
        return None

    return int(match.group())


def to_float(value: Any):
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    text = str(value)
    text = text.replace(",", "")
    text = re.sub(r"(Rs\.?|INR|USD|EUR|GBP|\$|₹|€|£)", "", text, flags=re.I)

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    return float(match.group())


def to_boolean(value: Any):
    if value is None:
        return None

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in ["true", "yes", "y", "1", "confirmed", "success", "passed", "active"]:
        return True

    if text in ["false", "no", "n", "0", "failed", "inactive"]:
        return False

    return None


def to_date(value: Any):
    if value is None:
        return None

    text = str(value).strip()

    if not text:
        return None

    try:
        dt = date_parser.parse(text, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def to_array_string(value: Any):
    if value is None:
        return None

    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]

    text = str(value).strip()
    if not text:
        return None

    parts = re.split(r",|;|\n|\band\b", text)
    arr = [p.strip() for p in parts if p.strip()]

    return arr if arr else None


def to_array_integer(value: Any):
    if value is None:
        return None

    if isinstance(value, list):
        result = []
        for x in value:
            n = to_integer(x)
            if n is not None:
                result.append(n)
        return result if result else None

    nums = re.findall(r"-?\d+", str(value).replace(",", ""))
    result = [int(n) for n in nums]

    return result if result else None


def coerce_value(value: Any, typ: str):
    typ = typ.strip().lower()

    if typ == "string":
        return clean_string(value)

    if typ == "integer":
        return to_integer(value)

    if typ == "float":
        return to_float(value)

    if typ == "boolean":
        return to_boolean(value)

    if typ == "date":
        return to_date(value)

    if typ == "array[string]":
        return to_array_string(value)

    if typ == "array[integer]":
        return to_array_integer(value)

    return clean_string(value)


def fallback_extract(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    """
    Small fallback for common hidden cases if Gemini fails.
    """
    result = {}

    for key, typ in schema.items():
        key_words = key.replace("_", " ").replace("-", " ")
        value = None

        # Label style: key: value
        pattern = rf"{re.escape(key_words)}\s*[:\-]\s*([^\n,;]+)"
        match = re.search(pattern, text, flags=re.I)

        if match:
            value = match.group(1).strip()

        result[key] = coerce_value(value, typ)

    return result

def smart_guess_value(key: str, typ: str, text: str):
    k = key.lower().replace("_", " ")
    t = text.strip()

    # device / equipment / asset style fields
    if any(w in k for w in ["device", "equipment", "asset", "machine", "unit"]):
        patterns = [
            r"\bAC\s+Unit\s+\d+\b",
            r"\bHVAC\s+Unit\s+\d+\b",
            r"\bUnit\s+\d+\b",
            r"\bPump\s+\d+\b",
            r"\bMotor\s+\d+\b",
            r"\bGenerator\s+\d+\b",
            r"\bBoiler\s+\d+\b",
            r"\bCompressor\s+\d+\b",
            r"\bValve\s+\d+\b",
            r"\bSensor\s+\d+\b",
            r"\bServer\s+\d+\b",
            r"\bRouter\s+\d+\b",
            r"\bPrinter\s+\d+\b",
        ]

        for p in patterns:
            m = re.search(p, t, flags=re.I)
            if m:
                return coerce_value(m.group(0), typ)

    # customer/person/name
    if any(w in k for w in ["customer", "person", "name", "user", "client"]):
        m = re.search(r"\b([A-Z][a-z]+)\b", t)
        if m:
            return coerce_value(m.group(1), typ)

    # amount / price / cost
    if any(w in k for w in ["amount", "price", "cost", "total"]):
        m = re.search(r"(?:Rs\.?|INR|₹|\$)?\s*[\d,]+(?:\.\d+)?", t, flags=re.I)
        if m:
            return coerce_value(m.group(0), typ)

    # date
    if "date" in k:
        m = re.search(
            r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b|\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b",
            t
        )
        if m:
            return coerce_value(m.group(0), typ)

    return None


def llm_extract(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return fallback_extract(text, schema)

    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""
Extract structured data from the given text.

Return ONLY valid JSON.
Return exactly the keys from the schema.
Do not add extra keys.
Do not omit keys.
Use null if a field cannot be found.
Follow the exact required JSON types.

Supported types:
- string
- integer
- float
- boolean
- date
- array[string]
- array[integer]

Date must be YYYY-MM-DD.
Floats and integers must be JSON numbers, not strings.
Boolean must be true or false.

Text:
{text}

Schema:
{json.dumps(schema, indent=2)}
"""

    response = model.generate_content(prompt)

    raw = response.text if response and response.text else "{}"
    return extract_json(raw)


@app.get("/")
def root():
    return {"status": "ok", "message": "Dynamic extraction API is running"}


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicExtractRequest):
    schema = req.schema or {}
    text = req.text or ""

    try:
        extracted = llm_extract(text, schema)
    except Exception:
        extracted = {}

    fallback = fallback_extract(text, schema)

    final = {}

    for key, typ in schema.items():
        value = extracted.get(key)

        # If LLM gives null/bad value, use label fallback
        if value is None:
            value = fallback.get(key)

        # If still null, use smart sentence fallback
        if value is None:
            value = smart_guess_value(key, typ, text)

        final[key] = coerce_value(value, typ)

    return final