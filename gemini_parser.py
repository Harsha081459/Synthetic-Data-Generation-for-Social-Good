"""
SynthoGen AI — Gemini API Parser
==================================
Sends a doctor's natural-language prompt to the Gemini REST API
and returns a structured JSON dict of patient constraints.

Uses the `requests` library (no Gemini SDK required — works on Python 3.6).
"""

import json
import os
import re
import requests


# ---------------------------------------------------------------------------
# .env loader (minimal, no third-party dependency)
# ---------------------------------------------------------------------------
def _load_env(path=None):
    """Read KEY=VALUE pairs from a .env file into os.environ."""
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


# Auto-load on import
_load_env()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_GROQ_MODEL = "llama-3.3-70b-versatile"
_GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

_SYSTEM_PROMPT = """You are a medical data parser. Given a doctor's natural language request for synthetic patient data, extract structured parameters and return ONLY a valid JSON object with these keys (omit any that are not mentioned):

{
  "num_patients": <integer, number of patients to generate>,
  "gender": <"male" or "female" or null if not specified>,
  "age_min": <integer or null>,
  "age_max": <integer or null>,
  "conditions": <list of strings, e.g. ["diabetes", "hypertension"]>,
  "severity": <"mild", "moderate", or "severe" or null>
}

Rules:
- Return ONLY the JSON object. No explanation, no markdown fences, no extra text.
- If a value is not mentioned, omit the key entirely.
- For conditions, use lowercase canonical names: "diabetes", "hypertension", "hypothyroidism", "asthma", "obesity".
- Default num_patients to 100 if not specified.
- "over age 50" means age_min=50. "under age 30" means age_max=30. "between 40 and 60" means age_min=40, age_max=60.
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_api_key():
    """Return the Groq API key or None. Checks env vars and Streamlit secrets."""
    key = os.environ.get("GROQ_API_KEY")
    if key:
        return key
    # Fallback: check Streamlit secrets (for Streamlit Cloud deployment)
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return None


def parse_prompt(prompt):
    """
    Send a doctor's prompt to Gemini and return a parsed dict.

    Returns:
        dict with keys like num_patients, gender, age_min, etc.

    Raises:
        ValueError  — if the API key is missing or the prompt is empty.
        RuntimeError — if the API call fails or the response is unparseable.
    """
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "Groq API key not found. "
            "Please create a .env file in the project root with:\n"
            "GROQ_API_KEY=your_key_here"
        )

    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("Prompt is empty. Please describe the patients you'd like to generate.")

    payload = {
        "model": _GROQ_MODEL,
        "messages": [
            {
                "role": "user",
                "content": _SYSTEM_PROMPT + "\n\nDoctor's request:\n" + prompt
            }
        ],
        "temperature": 0.1,
        "max_completion_tokens": 512,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.post(_GROQ_URL, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            return fallback_parse(prompt), True
            
        body = resp.json()
        text = body["choices"][0]["message"]["content"]
        return _extract_json(text), False
    except Exception as exc:
        return fallback_parse(prompt), True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_json(text):
    """Parse JSON from Gemini's text output, handling markdown fences."""
    text = text.strip()

    # Strip markdown code fences if present
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find the first { ... } block
        brace_match = re.search(r"\{[\s\S]*\}", text)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

    raise RuntimeError(
        "Could not parse Gemini response as JSON. Raw output:\n" + text[:500]
    )


def fallback_parse(prompt):
    """Offline regex-based fallback if Gemini API fails."""
    prompt_lower = prompt.lower()
    result = {}
    
    # Num patients
    num_match = re.search(r'\b(\d+)\b', prompt)
    if num_match:
        result["num_patients"] = int(num_match.group(1))
        
    # Gender
    if re.search(r'\b(male|men|man)\b', prompt_lower):
        result["gender"] = "male"
    elif re.search(r'\b(female|women|woman)\b', prompt_lower):
        result["gender"] = "female"
        
    # Age min/max
    age_min_match = re.search(r'(?:over|older than|>)\s*(?:age)?\s*(\d+)', prompt_lower)
    if age_min_match:
        result["age_min"] = int(age_min_match.group(1))
        
    age_max_match = re.search(r'(?:under|younger than|<)\s*(?:age)?\s*(\d+)', prompt_lower)
    if age_max_match:
        result["age_max"] = int(age_max_match.group(1))
        
    # Conditions
    conditions = []
    known_conditions = ["diabetes", "hypertension", "hypothyroidism", "asthma", "obesity"]
    for c in known_conditions:
        if c in prompt_lower:
            conditions.append(c)
    if conditions:
        result["conditions"] = conditions
        
    # Severity
    for s in ["mild", "moderate", "severe"]:
        if s in prompt_lower:
            result["severity"] = s
            break
            
    return result
