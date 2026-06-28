"""
SynthoGen AI — Prompt Parser (Validation & Normalization)
==========================================================
Takes the raw dict from Gemini and normalizes it into a clean
set of generation constraints ready for the patient generator.
"""


# ---------------------------------------------------------------------------
# Column mappings: condition name -> dataset column name + expected value
# ---------------------------------------------------------------------------
CONDITION_MAP = {
    "diabetes": ("Diabetes_Target", lambda v: v >= 1),
    "hypertension": ("Has_Hypertension", lambda v: v == 1),
    "hypothyroidism": ("Has_Hypothyroidism", lambda v: v == 1),
}

GENDER_MAP = {
    "male": 1,
    "m": 1,
    "female": 0,
    "f": 0,
}

SEVERITY_DIABETES = {
    "mild": 1,
    "moderate": 1,
    "severe": 2,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def validate_and_normalize(raw):
    """
    Validate and normalize a raw Gemini output dict.

    Parameters:
        raw (dict): Raw parsed JSON from Gemini.

    Returns:
        dict with normalized keys:
            - num_patients (int)
            - gender (int or None)
            - age_min (int or None)
            - age_max (int or None)
            - conditions (list of str)
            - severity (str or None)
            - condition_filters (list of (col_name, filter_fn) tuples)

    Raises:
        ValueError on invalid input.
    """
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object from Gemini, got: {}".format(type(raw).__name__))

    result = {}

    # --- num_patients ---
    num = raw.get("num_patients", 100)
    try:
        num = int(num)
    except (TypeError, ValueError):
        num = 100
    result["num_patients"] = max(1, min(num, 5000))

    # --- gender ---
    gender_raw = raw.get("gender")
    if gender_raw is not None:
        gender_raw = str(gender_raw).strip().lower()
        result["gender"] = GENDER_MAP.get(gender_raw)
        result["gender_label"] = gender_raw
    else:
        result["gender"] = None
        result["gender_label"] = None

    # --- age_min / age_max ---
    age_min = raw.get("age_min")
    age_max = raw.get("age_max")
    if age_min is not None:
        try:
            result["age_min"] = max(0, min(120, int(age_min)))
        except (TypeError, ValueError):
            result["age_min"] = None
    else:
        result["age_min"] = None

    if age_max is not None:
        try:
            result["age_max"] = max(0, min(120, int(age_max)))
        except (TypeError, ValueError):
            result["age_max"] = None
    else:
        result["age_max"] = None

    # --- conditions ---
    conditions = raw.get("conditions", [])
    if not isinstance(conditions, list):
        conditions = [str(conditions)]
    conditions = [str(c).strip().lower() for c in conditions]
    result["conditions"] = conditions

    # --- severity ---
    severity = raw.get("severity")
    if severity is not None:
        result["severity"] = str(severity).strip().lower()
    else:
        result["severity"] = None

    # --- Build column-level filters ---
    condition_filters = []
    for cond in conditions:
        if cond in CONDITION_MAP:
            col, fn = CONDITION_MAP[cond]
            condition_filters.append((cond, col, fn))
    result["condition_filters"] = condition_filters

    # Map severity to diabetes target value if applicable
    if result["severity"] and "diabetes" in conditions:
        target_val = SEVERITY_DIABETES.get(result["severity"])
        if target_val is not None:
            result["severity_filter"] = ("Diabetes_Target", target_val)
        else:
            result["severity_filter"] = None
    else:
        result["severity_filter"] = None

    return result


def format_constraints_summary(constraints):
    """Return a human-readable summary of the constraints for display."""
    lines = []
    lines.append("**Patients to generate:** {}".format(constraints["num_patients"]))

    if constraints.get("gender_label"):
        lines.append("**Gender:** {}".format(constraints["gender_label"].capitalize()))

    if constraints.get("age_min") is not None and constraints.get("age_max") is not None:
        lines.append("**Age range:** {} – {}".format(constraints["age_min"], constraints["age_max"]))
    elif constraints.get("age_min") is not None:
        lines.append("**Age:** ≥ {}".format(constraints["age_min"]))
    elif constraints.get("age_max") is not None:
        lines.append("**Age:** ≤ {}".format(constraints["age_max"]))

    if constraints.get("conditions"):
        lines.append("**Conditions:** {}".format(", ".join(c.capitalize() for c in constraints["conditions"])))

    if constraints.get("severity"):
        lines.append("**Severity:** {}".format(constraints["severity"].capitalize()))

    return "\n\n".join(lines)
