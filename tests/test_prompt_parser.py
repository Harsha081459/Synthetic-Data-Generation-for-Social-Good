"""
SynthoGen AI — Smoke Tests for Prompt Parser
==============================================
Tests the constraint validation and normalization logic
in prompt_parser.py without requiring any external APIs or data.
"""

import sys
import os

# Add project root to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_parser import validate_and_normalize, format_constraints_summary


class TestValidateAndNormalize:
    """Tests for the validate_and_normalize function."""

    def test_basic_defaults(self):
        """Empty input should return sensible defaults."""
        result = validate_and_normalize({})
        assert result["num_patients"] == 100
        assert result["gender"] is None
        assert result["age_min"] is None
        assert result["age_max"] is None
        assert result["conditions"] == []
        assert result["severity"] is None

    def test_num_patients_clamping(self):
        """num_patients should be clamped to [1, 10000]."""
        result = validate_and_normalize({"num_patients": 0})
        assert result["num_patients"] == 1

        result = validate_and_normalize({"num_patients": 99999})
        assert result["num_patients"] == 10000

        result = validate_and_normalize({"num_patients": 50})
        assert result["num_patients"] == 50

    def test_num_patients_invalid_type(self):
        """Non-numeric num_patients should fall back to 100."""
        result = validate_and_normalize({"num_patients": "abc"})
        assert result["num_patients"] == 100

    def test_gender_male(self):
        """Male gender should be normalized to 1."""
        result = validate_and_normalize({"gender": "male"})
        assert result["gender"] == 1
        assert result["gender_label"] == "male"

    def test_gender_female(self):
        """Female gender should be normalized to 0."""
        result = validate_and_normalize({"gender": "female"})
        assert result["gender"] == 0
        assert result["gender_label"] == "female"

    def test_gender_abbreviations(self):
        """Abbreviations m/f should be handled."""
        result = validate_and_normalize({"gender": "M"})
        assert result["gender"] == 1

        result = validate_and_normalize({"gender": "F"})
        assert result["gender"] == 0

    def test_gender_none(self):
        """Omitted gender should be None."""
        result = validate_and_normalize({"gender": None})
        assert result["gender"] is None

    def test_age_range(self):
        """Age min/max should be parsed and clamped to [0, 120]."""
        result = validate_and_normalize({"age_min": 30, "age_max": 60})
        assert result["age_min"] == 30
        assert result["age_max"] == 60

    def test_age_clamping(self):
        """Out-of-range ages should be clamped."""
        result = validate_and_normalize({"age_min": -5, "age_max": 200})
        assert result["age_min"] == 0
        assert result["age_max"] == 120

    def test_age_invalid_type(self):
        """Non-numeric ages should become None."""
        result = validate_and_normalize({"age_min": "old", "age_max": "young"})
        assert result["age_min"] is None
        assert result["age_max"] is None

    def test_conditions_list(self):
        """Conditions should be normalized to lowercase list."""
        result = validate_and_normalize({"conditions": ["Diabetes", "HYPERTENSION"]})
        assert result["conditions"] == ["diabetes", "hypertension"]

    def test_conditions_single_string(self):
        """A single string condition should be wrapped in a list."""
        result = validate_and_normalize({"conditions": "diabetes"})
        assert result["conditions"] == ["diabetes"]

    def test_condition_filters_built(self):
        """Known conditions should generate column-level filters."""
        result = validate_and_normalize({"conditions": ["diabetes", "hypertension"]})
        filter_names = [name for name, col, fn in result["condition_filters"]]
        assert "diabetes" in filter_names
        assert "hypertension" in filter_names

    def test_unknown_conditions_no_filter(self):
        """Unknown conditions should not generate filters but still be in the list."""
        result = validate_and_normalize({"conditions": ["asthma"]})
        assert "asthma" in result["conditions"]
        assert len(result["condition_filters"]) == 0

    def test_severity(self):
        """Severity should be normalized to lowercase."""
        result = validate_and_normalize({"severity": "Severe"})
        assert result["severity"] == "severe"

    def test_severity_filter_diabetes(self):
        """Severity + diabetes should produce a severity filter."""
        result = validate_and_normalize({
            "conditions": ["diabetes"],
            "severity": "severe"
        })
        assert result["severity_filter"] is not None
        assert result["severity_filter"][0] == "Diabetes_Target"
        assert result["severity_filter"][1] == 2

    def test_severity_filter_no_diabetes(self):
        """Severity without diabetes should not produce a severity filter."""
        result = validate_and_normalize({
            "conditions": ["hypertension"],
            "severity": "severe"
        })
        assert result["severity_filter"] is None

    def test_invalid_input_type(self):
        """Non-dict input should raise ValueError."""
        try:
            validate_and_normalize("not a dict")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_full_pipeline(self):
        """Test a realistic complete input."""
        raw = {
            "num_patients": 200,
            "gender": "female",
            "age_min": 40,
            "age_max": 65,
            "conditions": ["diabetes", "hypertension"],
            "severity": "moderate"
        }
        result = validate_and_normalize(raw)
        assert result["num_patients"] == 200
        assert result["gender"] == 0
        assert result["age_min"] == 40
        assert result["age_max"] == 65
        assert len(result["conditions"]) == 2
        assert len(result["condition_filters"]) == 2
        assert result["severity"] == "moderate"


class TestFormatConstraintsSummary:
    """Tests for the format_constraints_summary function."""

    def test_basic_summary(self):
        """Should include patient count."""
        constraints = validate_and_normalize({"num_patients": 50})
        summary = format_constraints_summary(constraints)
        assert "50" in summary

    def test_gender_in_summary(self):
        """Summary should mention gender when specified."""
        constraints = validate_and_normalize({"gender": "male"})
        summary = format_constraints_summary(constraints)
        assert "Male" in summary

    def test_age_range_in_summary(self):
        """Summary should show age range."""
        constraints = validate_and_normalize({"age_min": 30, "age_max": 50})
        summary = format_constraints_summary(constraints)
        assert "30" in summary
        assert "50" in summary

    def test_conditions_in_summary(self):
        """Summary should list conditions."""
        constraints = validate_and_normalize({"conditions": ["diabetes"]})
        summary = format_constraints_summary(constraints)
        assert "Diabetes" in summary

    def test_severity_in_summary(self):
        """Summary should show severity."""
        constraints = validate_and_normalize({"severity": "severe"})
        summary = format_constraints_summary(constraints)
        assert "Severe" in summary
