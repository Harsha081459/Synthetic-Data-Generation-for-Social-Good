"""
Tests for eval/evaluate.py — pure metric functions on toy DataFrames.

No model files, datasets, or network access required.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_spec = importlib.util.spec_from_file_location("eval_evaluate", ROOT / "eval" / "evaluate.py")
_evaluate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_evaluate)


class TestPrivacyDCR:
    def test_identical_rows_flagged_as_exact_matches(self):
        df_real = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
        df_synth = df_real.iloc[:2].copy()
        out = _evaluate.evaluate_privacy_dcr(df_real, df_synth)
        assert out["exact_match_count"] == 2
        assert out["exact_match_percent"] == 100.0
        assert out["min_dcr"] == 0.0

    def test_far_synthetic_rows_have_positive_dcr(self):
        df_real = pd.DataFrame({"a": [0.0, 1.0, 2.0], "b": [0.0, 1.0, 2.0]})
        df_synth = pd.DataFrame({"a": [100.0, 101.0], "b": [100.0, 101.0]})
        out = _evaluate.evaluate_privacy_dcr(df_real, df_synth)
        assert out["exact_match_count"] == 0
        assert out["avg_dcr"] > 0


def test_singleton_privacy_reference_is_supported():
    report = _evaluate.evaluate_privacy_dcr(pd.DataFrame({"a": [1.0]}), pd.DataFrame({"a": [1.0]}))
    assert report["exact_match_count"] == 1
    assert report["reference_rows_evaluated"] == 1


def test_diabetes_status_is_not_used_as_a_target_proxy():
    frame = pd.DataFrame({"Age": [20, 40], "Diabetes_Status": ["No", "Yes"], "Diabetes_Target": [0, 2]})
    real, synth = _evaluate.preprocess_for_ml(frame, frame.copy(), "Diabetes_Target")
    assert "Diabetes_Status" not in real and "Diabetes_Status" not in synth
    assert "Diabetes_Target" in real


class TestKAnonymity:
    def test_k_min_counts_smallest_group(self):
        df = pd.DataFrame(
            {
                "gender": ["M", "M", "F", "F", "F"],
                "blood": ["A", "B", "A", "B", "O"],
            }
        )
        out = _evaluate.evaluate_k_anonymity(df, ["gender", "blood"])
        # groups: (M,A)x2, (M,B)x1, (F,A)x1, (F,B)x1, (F,O)x1
        assert out["k_min"] == 1
        assert out["k_median"] == 1
        assert out["pct_records_k_leq_5"] == 100.0

    def test_uniform_groups(self):
        df = pd.DataFrame({"grp": ["x", "x", "y", "y"], "sub": ["p", "p", "q", "q"]})
        out = _evaluate.evaluate_k_anonymity(df, ["grp", "sub"])
        assert out["k_min"] == 2
        assert out["k_median"] == 2


class TestCorrelationSimilarity:
    def test_identical_frames_zero_mae(self):
        rng = np.random.RandomState(0)
        df = pd.DataFrame(rng.rand(30, 3), columns=["a", "b", "c"])
        out = _evaluate.evaluate_correlation_similarity(df, df.copy())
        assert out["corr_matrix_mae"] == 0.0

    def test_different_frames_positive_mae(self):
        rng = np.random.RandomState(0)
        df_real = pd.DataFrame(rng.rand(30, 3), columns=["a", "b", "c"])
        df_synth = pd.DataFrame(rng.rand(30, 3), columns=["a", "b", "c"])
        out = _evaluate.evaluate_correlation_similarity(df_real, df_synth)
        assert out["corr_matrix_mae"] > 0


class TestClassDistribution:
    def test_identical_distribution_zero_jsd(self):
        df = pd.DataFrame({"target": [0, 1, 0, 1, 0, 1]})
        out = _evaluate.evaluate_class_distribution(df, df.copy(), "target")
        assert out["jensen_shannon_divergence"] == pytest.approx(0.0, abs=1e-4)

    def test_missing_target_returns_none(self):
        df = pd.DataFrame({"x": [1, 2]})
        assert _evaluate.evaluate_class_distribution(df, df, "nope") is None
