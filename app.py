"""
SynthoGen AI — Streamlit Dashboard
====================================
A production-quality, multi-tab Streamlit app for:
  1. Data Explorer     – Load datasets, explore distributions, view merged data.
  2. Generative Engine – Train TVAE model, generate synthetic data.
  3. Utility Report    – SDMetrics scores, ML efficacy (XGBoost), distribution plots.
  4. Privacy Report    – Anonymeter attack simulations (singling-out, linkability,
                         inference risk).

Launch:   streamlit run app.py
"""

import os
import sys
import time
import logging

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- Add project root to path ----
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (
    load_diabetes,
    load_framingham,
    load_healthcare_monitoring,
    load_patient_readings,
    merge_datasets,
)
from src.synthesizer import SynthoGenEngine
from src.evaluate import (
    run_utility_evaluation,
    run_ml_efficacy,
    run_privacy_evaluation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Page Config & Custom CSS
# ============================================================================

st.set_page_config(
    page_title="SynthoGen AI | Privacy-Preserving Synthetic Data",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark-themed professional CSS
st.markdown("""
<style>
    /* ---- Global ---- */
    .main .block-container {
        padding-top: 1rem;
        max-width: 1200px;
    }

    /* ---- Metric cards ---- */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #a8b2d1 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #e6f1ff !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0a1a 0%, #1a1a2e 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #ccd6f6 !important;
    }

    /* ---- Tabs ---- */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #1a1a2e;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #8892b0;
        border: 1px solid #233554;
    }
    .stTabs [aria-selected="true"] {
        background: #233554 !important;
        color: #64ffda !important;
        border-bottom-color: #64ffda !important;
    }

    /* ---- Headers ---- */
    h1, h2, h3 {
        color: #ccd6f6 !important;
    }

    /* ---- Info boxes ---- */
    .success-box {
        background: linear-gradient(135deg, #0a2e0a 0%, #1a3a1a 100%);
        border: 1px solid #2d6a2d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .warning-box {
        background: linear-gradient(135deg, #2e2a0a 0%, #3a3a1a 100%);
        border: 1px solid #6a5a2d;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
    .info-box {
        background: linear-gradient(135deg, #0a1a2e 0%, #1a2a3e 100%);
        border: 1px solid #2d4a6a;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }

    /* ---- Footer ---- */
    .footer {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        padding: 20px 0;
        border-top: 1px solid #233554;
        margin-top: 40px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# Session State Initialisation
# ============================================================================

_DEFAULTS = {
    "diabetes_df": None,
    "framingham_df": None,
    "healthcare_df": None,
    "patient_df": None,
    "merged_df": None,
    "active_dataset_name": None,
    "active_dataset": None,
    "engine": None,
    "metadata_dict": None,
    "synthetic_data": None,
    "utility_results": None,
    "ml_results": None,
    "privacy_results": None,
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.markdown("## 🧬 SynthoGen AI")
st.sidebar.markdown("*Privacy-Preserving Synthetic Data for Social Good*")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Data Explorer",
        "⚙️ Generative Engine",
        "📈 Utility Report",
        "🔒 Privacy Report",
    ],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="footer">Built for DATAPORT Hackathon 2026</div>',
    unsafe_allow_html=True,
)


# ============================================================================
# Helper Functions
# ============================================================================

def _plotly_dark_template():
    """Return common plotly layout kwargs for dark theme."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(26,26,46,0.5)",
        font=dict(color="#ccd6f6"),
    )


def _render_metric_row(metrics: dict):
    """Render a row of st.metric cards from a dict."""
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        with col:
            if isinstance(value, float):
                st.metric(label, f"{value:.4f}")
            else:
                st.metric(label, str(value))


# ############################################################################
#  PAGE 1 — DATA EXPLORER
# ############################################################################

if page == "📊 Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("Load, inspect, and merge the three clinical datasets.")

    # ---- Dataset loading section ----
    st.subheader("1 · Load Datasets")

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("**🩺 Diabetes (MCDD)**")
        st.caption("16,353 rows · 19 clinical features")
        if st.button("Load Diabetes", key="btn_diab", use_container_width=True):
            with st.spinner("Loading Diabetes_MCDD.csv …"):
                st.session_state.diabetes_df = load_diabetes()
            st.success(f"✅ {len(st.session_state.diabetes_df):,} rows")

        if st.session_state.diabetes_df is not None:
            st.info(f"Shape: {st.session_state.diabetes_df.shape}")

    with col_b:
        st.markdown("**❤️ Framingham (CVD)**")
        st.caption("4,240 rows · 16 risk factors")
        if st.button("Load Framingham", key="btn_fram", use_container_width=True):
            with st.spinner("Loading Framingham.xlsx …"):
                st.session_state.framingham_df = load_framingham()
            st.success(f"✅ {len(st.session_state.framingham_df):,} rows")

        if st.session_state.framingham_df is not None:
            st.info(f"Shape: {st.session_state.framingham_df.shape}")

    with col_c:
        st.markdown("**🏥 Healthcare Mon.**")
        st.caption("10,000 rows · vitals")
        if st.button("Load Healthcare", key="btn_hmon", use_container_width=True):
            with st.spinner("Loading healthcare_monitoring_dataset.zip …"):
                st.session_state.healthcare_df = load_healthcare_monitoring()
            st.success(f"✅ {len(st.session_state.healthcare_df):,} rows")

        if st.session_state.healthcare_df is not None:
            st.info(f"Shape: {st.session_state.healthcare_df.shape}")

    with col_d:
        st.markdown("**📉 Patient Readings**")
        st.caption("23,351 rows · vitals")
        if st.button("Load Patient Rdgs", key="btn_pread", use_container_width=True):
            with st.spinner("Loading patient_readings...csv …"):
                st.session_state.patient_df = load_patient_readings()
            st.success(f"✅ {len(st.session_state.patient_df):,} rows")

        if st.session_state.patient_df is not None:
            st.info(f"Shape: {st.session_state.patient_df.shape}")

    # ---- Merge ----
    st.markdown("---")
    st.subheader("2 · Merge & Harmonize")
    st.markdown(
        "Combine loaded datasets into a unified schema with common clinical "
        "features: **Age, Sex, BMI, BP, Heart Rate, Glucose, Cholesterol**."
    )

    if st.button("🔗 Merge Datasets", use_container_width=True):
        loaded = {
            "diabetes_df": st.session_state.diabetes_df,
            "framingham_df": st.session_state.framingham_df,
            "healthcare_df": st.session_state.healthcare_df,
            "patient_df": st.session_state.patient_df,
        }
        loaded_clean = {k: v for k, v in loaded.items() if v is not None}

        if not loaded_clean:
            st.error("⚠️ Load at least one dataset first!")
        else:
            with st.spinner("Merging …"):
                st.session_state.merged_df = merge_datasets(**loaded_clean)
            st.success(
                f"✅ Merged dataset: {st.session_state.merged_df.shape[0]:,} rows × "
                f"{st.session_state.merged_df.shape[1]} cols"
            )

    # ---- Exploration ----
    st.markdown("---")
    st.subheader("3 · Explore")

    # Let user pick which dataset to explore / use for synthesis
    available = {}
    if st.session_state.diabetes_df is not None:
        available["Diabetes_MCDD"] = st.session_state.diabetes_df
    if st.session_state.framingham_df is not None:
        available["Framingham"] = st.session_state.framingham_df
    if st.session_state.healthcare_df is not None:
        available["Healthcare_Monitoring"] = st.session_state.healthcare_df
    if st.session_state.patient_df is not None:
        available["Patient_Readings"] = st.session_state.patient_df
    if st.session_state.merged_df is not None:
        available["Merged (Unified)"] = st.session_state.merged_df

    if available:
        choice = st.selectbox("Select dataset to explore & use for synthesis", list(available.keys()))
        df_view = available[choice]
        st.session_state.active_dataset_name = choice
        st.session_state.active_dataset = df_view

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df_view):,}")
        col2.metric("Columns", f"{len(df_view.columns)}")
        col3.metric("Numeric Cols", f"{len(df_view.select_dtypes(include=[np.number]).columns)}")
        col4.metric("NaN Cells", f"{df_view.isna().sum().sum():,}")

        # Data preview
        with st.expander("📋 Data Preview (first 50 rows)", expanded=True):
            st.dataframe(df_view.head(50), use_container_width=True)

        # Descriptive stats
        with st.expander("📊 Descriptive Statistics"):
            st.dataframe(df_view.describe().T, use_container_width=True)

        # Distribution plot
        st.markdown("#### Feature Distribution")
        num_cols = list(df_view.select_dtypes(include=[np.number]).columns)
        if num_cols:
            sel_col = st.selectbox("Pick a numeric column", num_cols, key="explore_col")
            fig = px.histogram(
                df_view, x=sel_col, nbins=50,
                title=f"Distribution of {sel_col}",
                color_discrete_sequence=["#64ffda"],
            )
            fig.update_layout(**_plotly_dark_template())
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        if len(num_cols) >= 2:
            with st.expander("🔥 Correlation Heatmap"):
                corr = df_view[num_cols].corr()
                fig_corr = px.imshow(
                    corr, text_auto=".2f",
                    color_continuous_scale="RdBu_r",
                    title="Feature Correlations",
                )
                fig_corr.update_layout(**_plotly_dark_template(), height=600)
                st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("👆 Load at least one dataset above to begin exploration.")


# ############################################################################
#  PAGE 2 — GENERATIVE ENGINE
# ############################################################################

elif page == "⚙️ Generative Engine":
    st.title("⚙️ Generative Engine — TVAE")
    st.markdown("Train a Tabular Variational Autoencoder and generate synthetic data.")

    if st.session_state.active_dataset is None:
        st.warning("⚠️ Please load and select a dataset in the **Data Explorer** tab first.")
        st.stop()

    df_active = st.session_state.active_dataset
    st.info(
        f"Active dataset: **{st.session_state.active_dataset_name}** "
        f"({len(df_active):,} rows × {len(df_active.columns)} cols)"
    )

    # ---- Hyper-parameters ----
    st.subheader("Training Configuration")
    col_hp1, col_hp2, col_hp3 = st.columns(3)

    with col_hp1:
        epochs = st.slider("Epochs", 50, 1000, 300, step=50)
        batch_size = st.select_slider("Batch Size", options=[64, 128, 256, 500, 1000], value=500)

    with col_hp2:
        embedding_dim = st.select_slider("Embedding Dim", options=[32, 64, 128, 256], value=128)
        max_train_rows = st.number_input(
            "Max training rows (0 = all)",
            min_value=0, max_value=len(df_active), value=min(5000, len(df_active)),
            step=500,
        )

    with col_hp3:
        num_synthetic = st.number_input(
            "Synthetic rows to generate",
            min_value=100, max_value=100000, value=min(len(df_active), 5000),
            step=500,
        )

    # ---- Training ----
    st.markdown("---")

    if st.button("🚀 Train TVAE & Generate", use_container_width=True, type="primary"):
        train_data = df_active.copy()

        # Sub-sample if requested
        if max_train_rows > 0 and max_train_rows < len(train_data):
            train_data = train_data.sample(max_train_rows, random_state=42)

        # Remove datetime columns (TVAE cannot handle them directly)
        datetime_cols = train_data.select_dtypes(include=["datetime64"]).columns.tolist()
        if datetime_cols:
            st.info(f"Dropping datetime columns for training: {datetime_cols}")
            train_data = train_data.drop(columns=datetime_cols)

        progress_bar = st.progress(0, text="Initialising SynthoGen Engine …")

        # Init engine
        engine = SynthoGenEngine(
            epochs=epochs,
            batch_size=batch_size,
            embedding_dim=embedding_dim,
        )

        progress_bar.progress(10, text="Detecting metadata …")
        time.sleep(0.3)

        progress_bar.progress(20, text=f"Training TVAE for {epochs} epochs …")

        try:
            engine.fit(train_data)
            progress_bar.progress(75, text="Generating synthetic data …")

            synthetic = engine.generate(num_rows=num_synthetic)
            progress_bar.progress(95, text="Saving model …")

            model_path = engine.save_model()

            progress_bar.progress(100, text="✅ Complete!")

            # Store in session
            st.session_state.engine = engine
            st.session_state.metadata_dict = engine.get_metadata_dict()
            st.session_state.synthetic_data = synthetic
            # Also store the training data for evaluation
            st.session_state.train_data = train_data

            st.balloons()
            st.success(
                f"Model trained & {len(synthetic):,} synthetic rows generated! "
                f"Model saved to `{model_path}`"
            )

        except Exception as e:
            progress_bar.empty()
            st.error(f"❌ Training failed: {e}")
            logger.exception("TVAE training error")
            st.stop()

    # ---- Show synthetic data if available ----
    if st.session_state.synthetic_data is not None:
        st.markdown("---")
        st.subheader("Generated Synthetic Data")

        syn = st.session_state.synthetic_data

        col1, col2, col3 = st.columns(3)
        col1.metric("Synthetic Rows", f"{len(syn):,}")
        col2.metric("Columns", f"{len(syn.columns)}")
        col3.metric("NaN Cells", f"{syn.isna().sum().sum()}")

        with st.expander("📋 Synthetic Data Preview", expanded=True):
            st.dataframe(syn.head(50), use_container_width=True)

        # Quick comparison histograms
        st.markdown("#### Quick Real vs Synthetic Comparison")
        num_cols_syn = list(syn.select_dtypes(include=[np.number]).columns)
        if num_cols_syn:
            cmp_col = st.selectbox("Select column", num_cols_syn, key="cmp_col")

            real_vals = st.session_state.train_data[cmp_col] if "train_data" in st.session_state and st.session_state.train_data is not None else st.session_state.active_dataset[cmp_col]
            syn_vals = syn[cmp_col]

            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=real_vals, name="Real", opacity=0.6,
                marker_color="#64ffda", nbinsx=50,
            ))
            fig.add_trace(go.Histogram(
                x=syn_vals, name="Synthetic", opacity=0.6,
                marker_color="#f77f00", nbinsx=50,
            ))
            fig.update_layout(
                barmode="overlay",
                title=f"{cmp_col} — Real vs Synthetic",
                **_plotly_dark_template(),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Download button
        csv_bytes = syn.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download Synthetic Data (CSV)",
            csv_bytes,
            file_name="synthogen_synthetic_data.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ############################################################################
#  PAGE 3 — UTILITY REPORT
# ############################################################################

elif page == "📈 Utility Report":
    st.title("📈 Utility Report")
    st.markdown("Evaluate how well the synthetic data preserves statistical properties and ML utility.")

    if st.session_state.synthetic_data is None:
        st.warning("⚠️ Generate synthetic data in the **Generative Engine** tab first.")
        st.stop()

    real = st.session_state.train_data if "train_data" in st.session_state and st.session_state.train_data is not None else st.session_state.active_dataset
    syn = st.session_state.synthetic_data
    metadata = st.session_state.metadata_dict

    # =========== SDMetrics Quality ===========
    st.subheader("1 · SDMetrics Quality Score")

    if st.session_state.utility_results is None:
        if st.button("🔬 Run Utility Evaluation", use_container_width=True, type="primary"):
            with st.spinner("Running SDMetrics evaluation …"):
                st.session_state.utility_results = run_utility_evaluation(real, syn, metadata)
            st.rerun()
    else:
        res = st.session_state.utility_results
        col1, col2, col3 = st.columns(3)

        with col1:
            score_pct = res["overall_score"] * 100
            st.metric("Overall Quality", f"{score_pct:.1f}%")
        with col2:
            if res["column_shapes"] is not None:
                st.metric("Column Shapes", f"{res['column_shapes']*100:.1f}%")
            else:
                st.metric("Column Shapes", "N/A")
        with col3:
            if res["column_pair_trends"] is not None:
                st.metric("Column Pair Trends", f"{res['column_pair_trends']*100:.1f}%")
            else:
                st.metric("Column Pair Trends", "N/A")

        # Quality gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score_pct,
            title={"text": "Overall Synthetic Data Quality", "font": {"color": "#ccd6f6"}},
            number={"suffix": "%", "font": {"color": "#64ffda"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#8892b0"},
                "bar": {"color": "#64ffda"},
                "bgcolor": "#1a1a2e",
                "steps": [
                    {"range": [0, 50], "color": "#e63946"},
                    {"range": [50, 75], "color": "#f4a261"},
                    {"range": [75, 90], "color": "#2a9d8f"},
                    {"range": [90, 100], "color": "#06d6a0"},
                ],
                "threshold": {
                    "line": {"color": "#e6f1ff", "width": 3},
                    "value": score_pct,
                },
            },
        ))
        fig_gauge.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#ccd6f6"},
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # =========== Distribution Overlap ===========
    st.markdown("---")
    st.subheader("2 · Distribution Overlap")

    common_num = sorted(
        set(real.select_dtypes(include=[np.number]).columns)
        & set(syn.select_dtypes(include=[np.number]).columns)
    )

    if common_num:
        # Show up to 6 columns in a grid
        show_cols = common_num[:6]
        n_plots = len(show_cols)
        n_plot_cols = min(3, n_plots)
        n_plot_rows = (n_plots + n_plot_cols - 1) // n_plot_cols

        fig_dist = make_subplots(
            rows=n_plot_rows, cols=n_plot_cols,
            subplot_titles=show_cols,
        )
        for i, c in enumerate(show_cols):
            row = i // n_plot_cols + 1
            col = i % n_plot_cols + 1
            fig_dist.add_trace(
                go.Histogram(x=real[c], name=f"Real", opacity=0.6,
                             marker_color="#64ffda", nbinsx=40, showlegend=(i == 0)),
                row=row, col=col,
            )
            fig_dist.add_trace(
                go.Histogram(x=syn[c], name=f"Synthetic", opacity=0.6,
                             marker_color="#f77f00", nbinsx=40, showlegend=(i == 0)),
                row=row, col=col,
            )

        fig_dist.update_layout(
            barmode="overlay",
            title="Real vs Synthetic Distributions",
            height=300 * n_plot_rows,
            **_plotly_dark_template(),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

    # =========== ML Efficacy ===========
    st.markdown("---")
    st.subheader("3 · ML Efficacy (XGBoost)")
    st.markdown(
        "Train XGBoost on **real** data and on **synthetic** data separately, "
        "then compare accuracy & F1 on a held-out real test set."
    )

    # Find potential target columns (categorical / low-cardinality integer)
    potential_targets = []
    for c in real.columns:
        if real[c].dtype == "object" or real[c].dtype.name == "category":
            potential_targets.append(c)
        elif real[c].nunique() <= 10 and c in syn.columns:
            potential_targets.append(c)

    if potential_targets:
        target_col = st.selectbox("Select target column for ML comparison", potential_targets)

        if st.session_state.ml_results is None:
            if st.button("🤖 Run ML Efficacy Test", use_container_width=True):
                with st.spinner("Training XGBoost models …"):
                    try:
                        st.session_state.ml_results = run_ml_efficacy(
                            real, syn, target_col
                        )
                        st.rerun()
                    except Exception as e:
                        st.error(f"ML Efficacy failed: {e}")
        else:
            ml = st.session_state.ml_results

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("##### 🟢 Trained on Real Data")
                st.metric("Accuracy", f"{ml['real_accuracy']:.4f}")
                st.metric("F1 Score (weighted)", f"{ml['real_f1']:.4f}")
                with st.expander("Classification Report"):
                    st.code(ml["real_report"])

            with col2:
                st.markdown("##### 🟠 Trained on Synthetic Data")
                st.metric("Accuracy", f"{ml['synth_accuracy']:.4f}",
                          delta=f"{ml['delta_accuracy']:+.4f}")
                st.metric("F1 Score (weighted)", f"{ml['synth_f1']:.4f}",
                          delta=f"{ml['delta_f1']:+.4f}")
                with st.expander("Classification Report"):
                    st.code(ml["synth_report"])

            # Bar chart comparison
            fig_ml = go.Figure(data=[
                go.Bar(name="Real", x=["Accuracy", "F1"],
                       y=[ml["real_accuracy"], ml["real_f1"]],
                       marker_color="#64ffda"),
                go.Bar(name="Synthetic", x=["Accuracy", "F1"],
                       y=[ml["synth_accuracy"], ml["synth_f1"]],
                       marker_color="#f77f00"),
            ])
            fig_ml.update_layout(
                barmode="group",
                title="ML Efficacy — Real vs Synthetic Training",
                yaxis_title="Score",
                **_plotly_dark_template(),
            )
            st.plotly_chart(fig_ml, use_container_width=True)
    else:
        st.info("No suitable target column found for ML comparison in this dataset.")

    # Reset button
    if st.button("🔄 Reset Evaluations"):
        st.session_state.utility_results = None
        st.session_state.ml_results = None
        st.rerun()


# ############################################################################
#  PAGE 4 — PRIVACY REPORT
# ############################################################################

elif page == "🔒 Privacy Report":
    st.title("🔒 Privacy Report — Anonymeter")
    st.markdown(
        "Simulate privacy attacks to measure re-identification risk: "
        "**singling-out**, **linkability**, and **inference**."
    )

    if st.session_state.synthetic_data is None:
        st.warning("⚠️ Generate synthetic data in the **Generative Engine** tab first.")
        st.stop()

    real = st.session_state.train_data if "train_data" in st.session_state and st.session_state.train_data is not None else st.session_state.active_dataset
    syn = st.session_state.synthetic_data

    st.markdown(
        '<div class="info-box">'
        "<b>How it works:</b> Anonymeter splits the real data 50/50 into a "
        "<i>training set</i> and a <i>control set</i>. It then runs "
        "privacy attacks against the synthetic data and compares the "
        "success rate to the control baseline. A <b>low risk score</b> "
        "means the synthetic data does NOT leak private information."
        "</div>",
        unsafe_allow_html=True,
    )

    n_attacks = st.slider("Number of attack attempts", 100, 2000, 500, step=100)

    if st.session_state.privacy_results is None:
        if st.button("🛡️ Run Privacy Evaluation", use_container_width=True, type="primary"):
            with st.spinner("Running anonymeter attacks … this may take a few minutes."):
                try:
                    st.session_state.privacy_results = run_privacy_evaluation(
                        real, syn, n_attacks=n_attacks,
                    )
                    st.rerun()
                except Exception as e:
                    st.error(f"Privacy evaluation failed: {e}")
                    logger.exception("Privacy eval error")
    else:
        priv = st.session_state.privacy_results

        # ---- Metric cards ----
        col1, col2, col3 = st.columns(3)

        def _format_risk(entry):
            if entry.get("risk") is not None:
                return f"{entry['risk']:.4f}"
            return f"Error: {entry.get('error', 'unknown')}"

        def _risk_color(entry):
            if entry.get("risk") is None:
                return "⚪"
            r = entry["risk"]
            if r < 0.05:
                return "🟢"
            elif r < 0.20:
                return "🟡"
            else:
                return "🔴"

        with col1:
            so = priv.get("singling_out", {})
            st.metric(
                f"{_risk_color(so)} Singling-Out Risk",
                _format_risk(so),
            )
            if so.get("ci"):
                st.caption(f"95% CI: [{so['ci'][0]:.4f}, {so['ci'][1]:.4f}]")

        with col2:
            lk = priv.get("linkability", {})
            st.metric(
                f"{_risk_color(lk)} Linkability Risk",
                _format_risk(lk),
            )
            if lk.get("ci"):
                st.caption(f"95% CI: [{lk['ci'][0]:.4f}, {lk['ci'][1]:.4f}]")

        with col3:
            inf = priv.get("inference", {})
            st.metric(
                f"{_risk_color(inf)} Inference Risk",
                _format_risk(inf),
            )
            if inf.get("ci"):
                st.caption(f"95% CI: [{inf['ci'][0]:.4f}, {inf['ci'][1]:.4f}]")
            if inf.get("secret_column"):
                st.caption(f"Secret: {inf['secret_column']}")

        # ---- Bar chart ----
        risk_names = []
        risk_vals = []
        risk_colors = []
        for name, key, color in [
            ("Singling-Out", "singling_out", "#e63946"),
            ("Linkability", "linkability", "#f4a261"),
            ("Inference", "inference", "#2a9d8f"),
        ]:
            entry = priv.get(key, {})
            if entry.get("risk") is not None:
                risk_names.append(name)
                risk_vals.append(entry["risk"])
                risk_colors.append(color)

        if risk_vals:
            fig_priv = go.Figure(data=[
                go.Bar(
                    x=risk_names, y=risk_vals,
                    marker_color=risk_colors,
                    text=[f"{v:.4f}" for v in risk_vals],
                    textposition="auto",
                )
            ])
            fig_priv.add_hline(
                y=0.05, line_dash="dash", line_color="#64ffda",
                annotation_text="Low-risk threshold (0.05)",
                annotation_font_color="#64ffda",
            )
            fig_priv.update_layout(
                title="Privacy Risk Scores (lower is better)",
                yaxis_title="Risk Score",
                yaxis_range=[0, max(0.5, max(risk_vals) * 1.2)],
                **_plotly_dark_template(),
            )
            st.plotly_chart(fig_priv, use_container_width=True)

        # ---- Detailed breakdown ----
        with st.expander("📊 Detailed Attack Results"):
            for name, key in [
                ("Singling-Out", "singling_out"),
                ("Linkability", "linkability"),
                ("Inference", "inference"),
            ]:
                entry = priv.get(key, {})
                st.markdown(f"**{name}**")
                if entry.get("error"):
                    st.error(f"Error: {entry['error']}")
                else:
                    detail_cols = st.columns(4)
                    detail_cols[0].metric("Risk", f"{entry.get('risk', 'N/A')}")
                    detail_cols[1].metric("Attack Rate", f"{entry.get('attack_rate', 'N/A')}")
                    detail_cols[2].metric("Baseline Rate", f"{entry.get('baseline_rate', 'N/A')}")
                    detail_cols[3].metric("Control Rate", f"{entry.get('control_rate', 'N/A')}")
                st.markdown("---")

        # ---- Interpretation ----
        st.markdown(
            '<div class="success-box">'
            "<b>Interpretation Guide:</b><br>"
            "🟢 <b>Risk &lt; 0.05</b> — Excellent privacy protection<br>"
            "🟡 <b>0.05 ≤ Risk &lt; 0.20</b> — Acceptable, monitor closely<br>"
            "🔴 <b>Risk ≥ 0.20</b> — Privacy concern, consider retraining with "
            "more noise or differential privacy"
            "</div>",
            unsafe_allow_html=True,
        )

    # Reset
    if st.button("🔄 Reset Privacy Evaluation"):
        st.session_state.privacy_results = None
        st.rerun()


# ============================================================================
# Footer
# ============================================================================
st.markdown(
    '<div class="footer">'
    "SynthoGen AI · Privacy-Preserving Synthetic Data for Social Good · DATAPORT Hackathon 2026 · "
    "Built with ❤️ using SDV, Anonymeter & Streamlit"
    "</div>",
    unsafe_allow_html=True,
)
