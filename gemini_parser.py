"""
SynthoGen AI - LLM Parser (Production)
========================================
Fallback chain:
  1. Groq API (Llama 3.3 70B) — primary
  2. xAI API (Grok) — secondary fallback
  3. Offline Regex Parser — final fallback
"""

import json
import os
import re
import requests


# ---------------------------------------------------------------------------
# .env loader
# ---------------------------------------------------------------------------
def _load_env(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(path):
        return
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())

_load_env()


# ---------------------------------------------------------------------------
# API Configurations
# ---------------------------------------------------------------------------
_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_XAI_MODEL = "grok-3-mini"
_XAI_URL = "https://api.x.ai/v1/chat/completions"

_SYSTEM_PROMPT = """You are a medical data parser. Given a doctor's natural language request for synthetic patient data, extract structured parameters and return ONLY a valid JSON object with these keys (omit any that are not mentioned):

{
  "num_patients": <integer, number of patients to generate>,
  "gender": <"male" or "female" or null if not specified>,
  "age_min": <integer or null>,
  "age_max": <integer or null>,
  "conditions": <list of strings, e.g. ["diabetes", "hypertension"]>,
  "severity": <"mild", "moderate", or "severe" or null>,
  "dataset": <"diabetes", "framingham", or "synthea" - infer from conditions>
}

Rules:
- Return ONLY the JSON object. No explanation, no markdown fences.
- If a value is not mentioned, omit the key entirely.
- For conditions, use lowercase: "diabetes", "hypertension", "hypothyroidism", "asthma", "obesity".
- Default num_patients to 100 if not specified.
- "over age 50" means age_min=50. "under 30" means age_max=30.
- If the request mentions heart disease/CHD/cardiovascular, set dataset to "framingham".
- If the request mentions EHR/electronic health records, set dataset to "synthea".
- Default dataset to "diabetes" if unclear.
"""


# ---------------------------------------------------------------------------
# Internal: Generic OpenAI-compatible API call
# ---------------------------------------------------------------------------
def _call_llm(api_url, api_key, model, prompt):
    """Make an OpenAI-compatible chat completion call. Raises on failure."""
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": _SYSTEM_PROMPT + "\n\nDoctor's request:\n" + prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 512,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(api_url, json=payload, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"API returned {resp.status_code}: {resp.text[:200]}")
    body = resp.json()
    text = body["choices"][0]["message"]["content"]
    return _extract_json(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_prompt(prompt):
    """
    Parse a doctor's prompt into structured constraints.

    Returns:
        (dict, bool) - (parsed constraints, True if offline fallback was used)

    Fallback chain: Groq (Llama 3.3) -> xAI (Grok) -> Offline regex.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is empty. Please describe the patients you'd like to generate.")

    # Try 1: Groq API (Llama 3.3 70B)
    key_groq = os.environ.get("GROQ_API_KEY")
    if key_groq:
        try:
            return _call_llm(_GROQ_URL, key_groq, _GROQ_MODEL, prompt), False
        except Exception:
            pass

    # Try 2: xAI API (Grok)
    key_xai = os.environ.get("XAI_API_KEY")
    if key_xai:
        try:
            return _call_llm(_XAI_URL, key_xai, _XAI_MODEL, prompt), False
        except Exception:
            pass

    # Try 3: Offline fallback
    return fallback_parse(prompt), True


# ---------------------------------------------------------------------------
# JSON Extraction
# ---------------------------------------------------------------------------
def _extract_json(text):
    """Parse JSON from LLM text output, handling markdown fences."""
    text = text.strip()
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass
    raise RuntimeError("Could not parse LLM response as JSON: " + text[:500])


# ---------------------------------------------------------------------------
# Offline Regex Fallback
# ---------------------------------------------------------------------------
def fallback_parse(prompt):
    """Offline regex-based parser when all API keys fail."""
    prompt_lower = prompt.lower()
    result = {}

    num_match = re.search(r'\b(\d+)\b', prompt)
    if num_match:
        result["num_patients"] = int(num_match.group(1))

    if re.search(r'\b(male|men|man)\b', prompt_lower):
        result["gender"] = "male"
    elif re.search(r'\b(female|women|woman)\b', prompt_lower):
        result["gender"] = "female"

    age_min_match = re.search(r'(?:over|older than|above|>)\s*(?:age)?\s*(\d+)', prompt_lower)
    if age_min_match:
        result["age_min"] = int(age_min_match.group(1))
    age_max_match = re.search(r'(?:under|younger than|below|<)\s*(?:age)?\s*(\d+)', prompt_lower)
    if age_max_match:
        result["age_max"] = int(age_max_match.group(1))

    between = re.search(r'between\s*(\d+)\s*and\s*(\d+)', prompt_lower)
    if between:
        result["age_min"] = int(between.group(1))
        result["age_max"] = int(between.group(2))

    conditions = []
    known = ["diabetes", "hypertension", "hypothyroidism", "asthma", "obesity",
             "heart disease", "cardiovascular", "prediabetes", "anemia"]
    for c in known:
        if c in prompt_lower:
            conditions.append(c.replace(" ", "_"))
    if conditions:
        result["conditions"] = conditions

    for s in ["mild", "moderate", "severe"]:
        if s in prompt_lower:
            result["severity"] = s
            break

    if any(c in prompt_lower for c in ["heart", "cardiovascular", "chd", "framingham"]):
        result["dataset"] = "framingham"
    elif any(c in prompt_lower for c in ["ehr", "synthea", "electronic"]):
        result["dataset"] = "synthea"
    else:
        result["dataset"] = "diabetes"

    return result
