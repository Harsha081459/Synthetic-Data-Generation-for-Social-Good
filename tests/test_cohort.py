import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from prompt_parser import filter_cohort, validate_and_normalize
import gemini_parser


def test_gender_and_condition_aliases_are_applied():
    data = pd.DataFrame({"gender": ["female", "F", "male"], "age": [40, 60, 40], "Cond_Essential_hypertension": [1.0, 0.0, 1.0]})
    constraints = validate_and_normalize({"gender": "female", "conditions": ["hypertension"], "age_max": 50})
    assert filter_cohort(data, constraints).index.tolist() == [0]


@pytest.mark.parametrize("conditions", [["asthma"], ["diabetes"]])
def test_unavailable_conditions_are_rejected_not_ignored(conditions):
    with pytest.raises(ValueError):
        filter_cohort(pd.DataFrame({"age": [30]}), validate_and_normalize({"conditions": conditions}))


def test_severity_is_not_ignored():
    data = pd.DataFrame({"Diabetes_Target": [0, 1, 2]})
    constraints = validate_and_normalize({"conditions": ["diabetes"], "severity": "severe"})
    assert filter_cohort(data, constraints).index.tolist() == [2]


def test_age_number_is_not_a_patient_count():
    assert "num_patients" not in gemini_parser.fallback_parse("patients over age 50")


def test_offline_negation_is_rejected():
    with pytest.raises(ValueError, match="negated"):
        gemini_parser.fallback_parse("Generate 10 patients without diabetes")


def test_missing_api_key_uses_offline_parser(monkeypatch):
    monkeypatch.setattr(gemini_parser, "get_api_key", lambda: None)
    raw, fallback = gemini_parser.parse_prompt("Generate 5 female patients with diabetes")
    assert fallback and raw["num_patients"] == 5 and raw["gender"] == "female"
