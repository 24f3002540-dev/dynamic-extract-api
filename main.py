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
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

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
    text = str(text).strip()
    text = text.replace("```json", "").replace("```", "").strip()

    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return {}


def clean_string(value: Any):
    if value is None:
        return None
    value = str(value).strip().strip('"').strip("'")
    if value.lower() in ["null", "none", "not found", "unknown", ""]:
        return None
    return value


def to_integer(value: Any):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)

    text = str(value).replace(",", "")
    m = re.search(r"-?\d+", text)
    return int(m.group()) if m else None


def to_float(value: Any):
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).replace(",", "")
    text = re.sub(r"(Rs\.?|INR|USD|EUR|GBP|\$|₹|€|£)", "", text, flags=re.I)
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(m.group()) if m else None


def to_boolean(value: Any):
    if value is None:
        return None
    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()
    true_words = ["true", "yes", "y", "1", "confirmed", "success", "passed", "active", "available", "enabled"]
    false_words = ["false", "no", "n", "0", "failed", "inactive", "unavailable", "disabled"]

    if text in true_words:
        return True
    if text in false_words:
        return False
    return None


def to_date(value: Any):
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    try:
        dt = date_parser.parse(text, dayfirst=True, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def to_array_string(value: Any):
    if value is None:
        return None

    if isinstance(value, list):
        arr = [clean_string(x) for x in value]
        arr = [x for x in arr if x]
        return arr if arr else None

    text = str(value).strip()
    if not text:
        return None

    text = re.sub(r"^\[|\]$", "", text)
    parts = re.split(r",|;|\n|\band\b", text, flags=re.I)
    arr = [p.strip().strip('"').strip("'") for p in parts if p.strip()]
    return arr if arr else None


def to_array_integer(value: Any):
    if value is None:
        return None

    if isinstance(value, list):
        arr = []
        for x in value:
            n = to_integer(x)
            if n is not None:
                arr.append(n)
        return arr if arr else None

    nums = re.findall(r"-?\d+", str(value).replace(",", ""))
    arr = [int(x) for x in nums]
    return arr if arr else None


def coerce_value(value: Any, typ: str):
    typ = str(typ).strip().lower()

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


def label_extract(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    result = {}

    for key, typ in schema.items():
        key_words = key.replace("_", " ").replace("-", " ")
        candidates = [
            key,
            key_words,
            key_words.title(),
            key_words.capitalize(),
        ]

        value = None

        for name in candidates:
            patterns = [
                rf"\b{re.escape(name)}\b\s*[:\-]\s*([^\n.;]+)",
                rf"\b{re.escape(name)}\b\s+is\s+([^\n.;]+)",
                rf"\b{re.escape(name)}\b\s+was\s+([^\n.;]+)",
            ]

            for p in patterns:
                m = re.search(p, text, flags=re.I)
                if m:
                    value = m.group(1).strip()
                    break

            if value is not None:
                break

        result[key] = coerce_value(value, typ)

    return result


def smart_guess_value(key: str, typ: str, text: str):
    k = key.lower().replace("_", " ").replace("-", " ")
    t = text.strip()

    # title / paper / article / book
    if any(w in k for w in ["title", "paper", "article", "report", "document", "book"]):
        patterns = [
            r'(?:title|titled|called|named)\s*(?:is|as|:)?\s*["“”\']([^"“”\']+)["“”\']',
            r'(?:paper|article|report|document|book)\s+(?:titled|called|named)\s*["“”\']?(.+?)(?:["“”\']| by | authored | written | published | on |,|\.|$)',
            r'\btitle\s*[:\-]\s*([^\n.;]+)',
        ]
        for p in patterns:
            m = re.search(p, t, flags=re.I)
            if m:
                return coerce_value(m.group(1).strip(), typ)

    # device / equipment
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

    # date fields
    if "date" in k:
        patterns = [
            r"\b\d{4}-\d{2}-\d{2}\b",
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{4}\b",
            r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
            r"\b[A-Za-z]+\s+\d{1,2},\s*\d{4}\b",
        ]
        for p in patterns:
            m = re.search(p, t)
            if m:
                return coerce_value(m.group(0), typ)

    # time fields as string
    if "time" in k and typ.lower() == "string":
        m = re.search(r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b", t)
        if m:
            return coerce_value(m.group(0), typ)

    # amount / price / cost
    if any(w in k for w in ["amount", "price", "cost", "total", "fee", "salary", "revenue"]):
        m = re.search(r"(?:Rs\.?|INR|₹|\$|USD)?\s*\d[\d,]*(?:\.\d+)?", t, flags=re.I)
        if m:
            return coerce_value(m.group(0), typ)

    # quantity / count
    if any(w in k for w in ["quantity", "count", "number", "units"]):
        m = re.search(r"\b\d+\b", t)
        if m:
            return coerce_value(m.group(0), typ)

    # customer/person/name
    if any(w in k for w in ["customer", "person", "client", "user", "buyer"]):
        m = re.search(r"\b([A-Z][a-z]+)\b", t)
        if m:
            return coerce_value(m.group(1), typ)

    # store/vendor/company
    if any(w in k for w in ["store", "vendor", "company", "seller", "supplier"]):
        patterns = [
            r"\bfrom\s+([A-Z][A-Za-z0-9& ]+?)(?:\.|,|$)",
            r"\bat\s+([A-Z][A-Za-z0-9& ]+?)(?:\.|,|$)",
        ]
        for p in patterns:
            m = re.search(p, t)
            if m:
                return coerce_value(m.group(1).strip(), typ)

    return None


def fallback_extract(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    labelled = label_extract(text, schema)
    result = {}

    for key, typ in schema.items():
        value = labelled.get(key)

        if value is None:
            value = smart_guess_value(key, typ, text)

        result[key] = coerce_value(value, typ)

    return result


def llm_extract(text: str, schema: Dict[str, str]) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {}

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)

        prompt = f"""
Extract structured data from text.

Return ONLY valid JSON.
Return exactly these schema keys.
No extra keys.
No missing keys.
Use null when not found.
Dates must be YYYY-MM-DD.
Numbers must be JSON numbers.
Booleans must be true/false.
Arrays must be JSON arrays.

Text:
{text}

Schema:
{json.dumps(schema, indent=2)}
"""

        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0,
                "max_output_tokens": 512,
            },
            request_options={"timeout": 18},
        )

        raw = response.text if response and response.text else "{}"
        return extract_json(raw)

    except Exception:
        return {}


@app.get("/")
def root():
    return {"status": "ok"}


@app.post("/dynamic-extract")
def dynamic_extract(req: DynamicExtractRequest):
    schema = req.schema or {}
    text = req.text or ""

    extracted = llm_extract(text, schema)
    fallback = fallback_extract(text, schema)

    final = {}

    for key, typ in schema.items():
        value = extracted.get(key)

        if value is None:
            value = fallback.get(key)

        final[key] = coerce_value(value, typ)

    return final