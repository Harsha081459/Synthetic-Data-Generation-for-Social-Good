from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import gemini_parser

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dashboard(monkeypatch):
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(gemini_parser, "get_api_key", lambda: None)
    def offline(*args, **kwargs):
        raise AssertionError("Dashboard tests must not call an external API")
    monkeypatch.setattr(gemini_parser.requests, "post", offline)
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()


def test_all_pages_render_for_each_dataset(dashboard):
    for dataset in dashboard.sidebar.selectbox[0].options:
        dashboard.sidebar.selectbox[0].set_value(dataset).run()
        for page in dashboard.radio[0].options:
            dashboard.radio[0].set_value(page).run()
            assert not dashboard.exception, [exc.message for exc in dashboard.exception]


@pytest.mark.parametrize("model", ["tvae", "ctgan", "tabddpm", "tabsyn"])
def test_generator_labels_cached_output_truthfully(dashboard, model):
    dashboard.radio[0].set_value(dashboard.radio[0].options[-1]).run()
    dashboard.selectbox(key="man_model").set_value(model).run()
    dashboard.number_input(key="man_num").set_value(10).run()
    dashboard.button(key="man_btn").click().run()
    assert not dashboard.exception
    assert dashboard.dataframe[0].value.shape[0] == 10
    assert any(metric.value == "Cached model output" for metric in dashboard.metric)
    pool = pd.read_csv(ROOT / f"data/synthetic/{model}_diabetes.csv")
    result = dashboard.dataframe[0].value
    assert pd.MultiIndex.from_frame(result).isin(pd.MultiIndex.from_frame(pool)).all()


def test_prompt_generation_without_api_key(dashboard):
    dashboard.radio[0].set_value(dashboard.radio[0].options[-1]).run()
    dashboard.text_area(key="p2p_prompt").input("Generate 5 female patients over age 30 with diabetes").run()
    dashboard.button(key="p2p_btn").click().run()
    assert not dashboard.exception
    assert not dashboard.error, [error.value for error in dashboard.error]
    data = dashboard.dataframe[-1].value
    assert len(data) == 5
    assert (data["Sex"] == 0).all()
    assert (data["Age"] >= 30).all()
    assert (data["Diabetes_Target"] >= 1).all()
