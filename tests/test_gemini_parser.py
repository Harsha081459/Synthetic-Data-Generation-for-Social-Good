"""
SynthoGen AI — Smoke Tests for Gemini Parser (Fallback Mode)
==============================================================
Tests the offline regex-based fallback parser in gemini_parser.py.
These tests don't require any API keys or network access.
"""

import sys
import os

# Add project root to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gemini_parser import fallback_parse, _extract_json


class TestFallbackParse:
    """Tests for the offline regex-based fallback parser."""

    def test_num_patients_extraction(self):
        """Should extract numeric patient count."""
        result = fallback_parse("Generate 200 patients with diabetes")
        assert result["num_patients"] == 200

    def test_gender_male(self):
        """Should detect male gender."""
        result = fallback_parse("50 male patients over age 40")
        assert result["gender"] == "male"

    def test_gender_female(self):
        """Should detect female gender."""
        result = fallback_parse("Generate 100 female patients")
        assert result["gender"] == "female"

    def test_gender_women(self):
        """Should detect 'women' as female."""
        result = fallback_parse("200 women with hypertension")
        assert result["gender"] == "female"

    def test_age_min_over(self):
        """'over age X' should set age_min."""
        result = fallback_parse("patients over age 50")
        assert result["age_min"] == 50

    def test_age_max_under(self):
        """'under age X' should set age_max."""
        result = fallback_parse("patients under age 30")
        assert result["age_max"] == 30

    def test_age_range_between(self):
        """'between X and Y' should set both age_min and age_max."""
        result = fallback_parse("patients between 40 and 60")
        assert result["age_min"] == 40
        assert result["age_max"] == 60

    def test_age_range_aged_to(self):
        """'aged X to Y' should set both age_min and age_max."""
        result = fallback_parse("50 patients aged 30 to 50")
        assert result["age_min"] == 30
        assert result["age_max"] == 50

    def test_age_range_years(self):
        """'X-Y years old' should set both age_min and age_max."""
        result = fallback_parse("patients 20-40 years old with diabetes")
        assert result["age_min"] == 20
        assert result["age_max"] == 40

    def test_condition_diabetes(self):
        """Should detect diabetes condition."""
        result = fallback_parse("100 patients with diabetes")
        assert "diabetes" in result["conditions"]

    def test_condition_hypertension(self):
        """Should detect hypertension condition."""
        result = fallback_parse("patients with hypertension")
        assert "hypertension" in result["conditions"]

    def test_multiple_conditions(self):
        """Should detect multiple conditions."""
        result = fallback_parse("patients with diabetes and hypertension")
        assert "diabetes" in result["conditions"]
        assert "hypertension" in result["conditions"]

    def test_severity(self):
        """Should detect severity levels."""
        result = fallback_parse("patients with severe diabetes")
        assert result["severity"] == "severe"

    def test_severity_mild(self):
        """Should detect mild severity."""
        result = fallback_parse("mild diabetes patients")
        assert result["severity"] == "mild"

    def test_no_gender(self):
        """Should not set gender when not mentioned."""
        result = fallback_parse("100 patients with diabetes")
        assert "gender" not in result

    def test_no_conditions(self):
        """Should not set conditions when none mentioned."""
        result = fallback_parse("100 patients over age 50")
        assert "conditions" not in result

    def test_complex_prompt(self):
        """Full complex prompt should extract all fields."""
        result = fallback_parse(
            "Generate 500 female patients between 40 and 65 "
            "with severe diabetes and hypertension"
        )
        assert result["num_patients"] == 500
        assert result["gender"] == "female"
        assert result["age_min"] == 40
        assert result["age_max"] == 65
        assert "diabetes" in result["conditions"]
        assert "hypertension" in result["conditions"]
        assert result["severity"] == "severe"

    def test_empty_prompt(self):
        """Empty prompt should return empty dict."""
        result = fallback_parse("")
        assert isinstance(result, dict)


class TestExtractJson:
    """Tests for the JSON extraction helper."""

    def test_plain_json(self):
        """Should parse plain JSON string."""
        result = _extract_json('{"num_patients": 100}')
        assert result["num_patients"] == 100

    def test_json_with_markdown_fences(self):
        """Should handle JSON wrapped in ```json ... ``` fences."""
        text = '```json\n{"num_patients": 200, "gender": "male"}\n```'
        result = _extract_json(text)
        assert result["num_patients"] == 200
        assert result["gender"] == "male"

    def test_json_with_surrounding_text(self):
        """Should extract JSON from surrounding prose."""
        text = 'Here is the result: {"num_patients": 50} Hope this helps!'
        result = _extract_json(text)
        assert result["num_patients"] == 50

    def test_invalid_json_raises(self):
        """Non-JSON text should raise RuntimeError."""
        try:
            _extract_json("This is not JSON at all")
            assert False, "Should have raised RuntimeError"
        except RuntimeError:
            pass
