
import os
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except Exception:
    GENAI_AVAILABLE = False

st.set_page_config(
    page_title="🏭 Manufacturing Defect Prediction – Executive Dashboard",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)
# -------------------- CONSTANTS --------------------
FEATURES = ["ProductionVolume", "SupplierQuality", "MaintenanceHours", "EnergyEfficiency"]
DEFAULT_MODEL_PATH = r"C:\\Users\\Harish.Kumar\\Manufacturing_defect_prediction_app\\Manufacturing_Defect_Prediction-\\logistic_regression_model.pkl"
DEFAULT_SCALER_PATH = r"C:\\Users\\Harish.Kumar\\Manufacturing_defect_prediction_app\\Manufacturing_Defect_Prediction-\\standard_scaler.pkl"
SAMPLE_DATA_PATH = "/mnt/data/customized_manufacturing_dataset.csv"  # used only if available in this session

# -------------------- THEME / STYLES --------------------
CUSTOM_CSS = """
<style>
/***** Global tweaks *****/
section.main > div { padding-top: 0.5rem; }

/***** Headings *****/
h1, h2, h3 { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; }

/***** Cards *****/
.block-card { background: #ffffff; border: 1px solid #edf2f7; border-radius: 14px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }

/***** Sidebar *****/
[data-testid="stSidebar"] { background-color: #0f172a; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color: #e5e7eb !important; }

/***** Metrics row *****/
.metric-card { background: #0ea5e9; color: white; border-radius: 12px; padding: 12px 14px; }
.metric-card h3 { margin: 0; font-size: 14px; opacity: 0.9; }
.metric-card .val { font-size: 26px; font-weight: 700; margin-top: 6px; }

</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png")
st.sidebar.title("Defect Prediction")
st.sidebar.caption("Interactive, explainable, business-ready")

# Model + Scaler configuration
st.sidebar.subheader("⚙️ Model Configuration")
model_path_in = st.sidebar.text_input("Model path (.pkl)", value=DEFAULT_MODEL_PATH)
scaler_path_in = st.sidebar.text_input("Scaler path (.pkl)", value=DEFAULT_SCALER_PATH)

st.sidebar.markdown("**Or upload model/scaler** (overrides paths)")
uploaded_model = st.sidebar.file_uploader("Upload model (joblib .pkl)", type=["pkl"], key="mdl")
uploaded_scaler = st.sidebar.file_uploader("Upload scaler (joblib .pkl)", type=["pkl"], key="scl")

# AI Settings
st.sidebar.subheader("🧠 AI Settings (optional)")
api_key_env = os.getenv("GOOGLE_API_KEY", "")
api_key_ui = st.sidebar.text_input("Google API Key", type="password", value=api_key_env)
if api_key_ui and GENAI_AVAILABLE:
    try:
        genai.configure(api_key=api_key_ui)
        ai_ready = True
    except Exception:
        ai_ready = False
else:
    ai_ready = False
st.sidebar.caption("AI summaries will auto-enable when a valid key is provided.")

st.sidebar.markdown("---")
st.sidebar.markdown("**Developed by:** Harish Kumar")

# -------------------- HELPERS --------------------
@st.cache_resource(show_spinner=False)
def load_joblib_from_path_or_bytes(path: str, uploaded) -> Optional[object]:
    try:
        if uploaded is not None:
            return joblib.load(uploaded)
        if path and os.path.exists(path):
            return joblib.load(path)
    except Exception as e:
        st.sidebar.error(f"Failed to load file: {e}")
    return None

@st.cache_data(show_spinner=False)
def read_dataset(file) -> pd.DataFrame:
    if file is None:
        # Try sample data if present
        if os.path.exists(SAMPLE_DATA_PATH):
            return pd.read_csv(SAMPLE_DATA_PATH)
        return pd.DataFrame()
    name = getattr(file, 'name', 'uploaded')
    if name.lower().endswith('.csv'):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


def ensure_feature_order(df: pd.DataFrame, features: list[str]) -> Tuple[pd.DataFrame, list[str]]:
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    return df[present].copy(), missing


def sanity_check_predictions(y_pred: np.ndarray) -> Optional[str]:
    # Warn if predictions are all one class
    if len(np.unique(y_pred)) == 1:
        cls = int(np.unique(y_pred)[0])
        return f"All predictions are class {cls}. This often indicates a scaling or feature mismatch. Make sure you're using the SAME StandardScaler from training."
    return None


def explain_prediction_logreg(model, scaler, X_df_row: pd.DataFrame) -> pd.DataFrame:
    """Return per-feature contribution for a single row using logistic regression.
    Requires scaler fitted on training data for meaningful contributions.
    """
    coefs = model.coef_.reshape(-1)
    intercept = float(model.intercept_[0]) if hasattr(model, 'intercept_') else 0.0

    # Use scaler if provided; else use raw (less meaningful)
    if scaler is not None:
        x_scaled = scaler.transform(X_df_row.values)
        x_used = x_scaled.reshape(-1)
        scale_note = "(using trained scaler)"
    else:
        x_used = X_df_row.values.reshape(-1)
        scale_note = "(raw values — provide scaler for accurate attribution)"

    contrib = coefs * x_used
    total_logit = float(np.sum(contrib) + intercept)
    prob = float(1 / (1 + np.exp(-total_logit)))
    out = pd.DataFrame({
        "Feature": X_df_row.columns,
        "Value": X_df_row.values.reshape(-1),
        "Contribution": contrib,
    }).sort_values("Contribution", ascending=False)
    out.attrs["scale_note"] = scale_note
    out.attrs["intercept"] = intercept
    out.attrs["logit"] = total_logit
    out.attrs["prob"] = prob
    return out

# -------------------- PAGE TITLE --------------------
st.title("🏭 Manufacturing Defect Prediction – Executive Dashboard")
st.caption("Understand your data, predict defects, and explain outcomes with confidence.")

# -------------------- TABS --------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 EDA",
    "🤖 ML Model",
    "✨ AI Enhancements",
    "📌 Final Insights",
])

# -------------------- LOAD MODEL & SCALER (shared) --------------------
model = load_joblib_from_path_or_bytes(model_path_in, uploaded_model)
scaler = load_joblib_from_path_or_bytes(scaler_path_in, uploaded_scaler)

if model is None:
    st.error("❌ Model not loaded. Please provide a valid path or upload the trained model.")
else:
    if not hasattr(model, "predict"):
        st.error("❌ Loaded object does not look like a scikit-learn model (no .predict).")
    elif getattr(model, "n_features_in_", None) not in (None, len(FEATURES)):
        st.warning(f"ℹ️ Model reports n_features_in_={getattr(model, 'n_features_in_', 'unknown')}. Expected {len(FEATURES)} for {FEATURES}.")

if model is not None and scaler is None:
    st.warning("⚠️ No scaler loaded. If your model was trained on scaled inputs, predictions may be wrong. Upload or set the path for the training StandardScaler.")

# ======================= TAB 1: EDA =======================
with tab1:
    st.subheader("Data Upload / Load")
    data_file = st.file_uploader("Upload CSV/Excel", type=["csv", "xlsx"], help="Include 'DefectStatus' column if available.")
    df = read_dataset(data_file)

    if df.empty:
        st.info("Upload a dataset to begin. If you're running locally with a sample file, it will auto-load when available.")
    else:
        # ------- Data Preview -------
        st.markdown("### 🔍 Data Preview")
        st.dataframe(df.head(20), use_container_width=True)

        # ------- Data Quality -------
        st.markdown("### 🧹 Data Quality Check")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write("**Rows**", df.shape[0])
            st.write("**Columns**", df.shape[1])
        with col_b:
            missing = df.isna().sum()
            st.write("**Missing values**")
            st.bar_chart(missing)
        with col_c:
            st.write("**Duplicates**")
            st.write(int(df.duplicated().sum()))

        # ------- Descriptive Stats -------
        st.markdown("### 📈 Descriptive Statistics")
        st.dataframe(df.describe(include='all').transpose(), use_container_width=True)

        # ------- Filters -------
        st.markdown("### 🔎 Interactive Filters")
        fcols = [c for c in FEATURES if c in df.columns]
        if fcols:
            with st.expander("Filter rows by ranges"):
                fdf = df.copy()
                for c in fcols:
                    cmin, cmax = float(fdf[c].min()), float(fdf[c].max())
                    lo, hi = st.slider(f"{c} range", min_value=cmin, max_value=cmax, value=(cmin, cmax))
                    fdf = fdf[(fdf[c] >= lo) & (fdf[c] <= hi)]
                st.write(f"Filtered rows: {len(fdf)}")
                st.dataframe(fdf.head(10))

        # ------- Visualizations -------
        st.markdown("### 📊 Visualizations")
        v1, v2 = st.columns(2)
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            with v1:
                sel = st.selectbox("Distribution of", options=num_cols, index=0)
                fig, ax = plt.subplots()
                sns.histplot(df[sel].dropna(), kde=True, ax=ax)
                ax.set_title(f"Distribution: {sel}")
                st.pyplot(fig)
            with v2:
                if len(num_cols) > 1:
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
                    ax.set_title("Correlation Heatmap")
                    st.pyplot(fig)

# ======================= TAB 2: ML MODEL =======================
with tab2:
    st.subheader("Interactive Prediction (Manual & Batch)")

    # Manual prediction UI uses true business features and recommended ranges
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Manual Input")
        st.caption("Use the typical training ranges: ProductionVolume [100–999], SupplierQuality [80–100], MaintenanceHours [0–23], EnergyEfficiency [0.1–0.5]")
        pv = st.number_input("🏭 Production Volume", min_value=100, max_value=999, value=500, step=1)
        sq = st.number_input("📦 Supplier Quality", min_value=80.0, max_value=100.0, value=90.0, step=0.1, format="%.1f")
        mh = st.number_input("🛠️ Maintenance Hours", min_value=0, max_value=23, value=8, step=1)
        ee = st.number_input("⚡ Energy Efficiency", min_value=0.1, max_value=0.5, value=0.3, step=0.01, format="%.2f")
        manual_df = pd.DataFrame([[pv, sq, mh, ee]], columns=FEATURES)
        st.dataframe(manual_df, use_container_width=True)

        pred_btn = st.button("🚀 Predict")
    with col2:
        st.markdown("#### Batch Prediction (CSV)")
        batch_file = st.file_uploader("Upload CSV with columns: ProductionVolume, SupplierQuality, MaintenanceHours, EnergyEfficiency", type=["csv"], key="batch")

    def predict_df(_model, _scaler, Xdf: pd.DataFrame) -> Tuple[np.ndarray, Optional[str]]:
        if not all(c in Xdf.columns for c in FEATURES):
            return None, "CSV missing required columns."
        X = Xdf[FEATURES].copy()
        try:
            if _scaler is not None:
                Xs = _scaler.transform(X)
                yhat = _model.predict(Xs)
                return yhat, sanity_check_predictions(yhat)
            else:
                # No scaler; try raw (but warn)
                yhat = _model.predict(X)
                warn = "No scaler used; predictions may be inaccurate."
                return yhat, warn
        except Exception as e:
            return None, f"Prediction error: {e}"

    # Manual predict flow
    if pred_btn and model is not None:
        yhat, warn = predict_df(model, scaler, manual_df)
        if yhat is None:
            st.error(warn)
        else:
            pred = int(yhat[0])
            prob = None
            try:
                if scaler is not None:
                    prob = float(model.predict_proba(scaler.transform(manual_df))[0, 1])
                else:
                    prob = float(model.predict_proba(manual_df)[0, 1])
            except Exception:
                pass

            if pred == 1:
                st.error(f"Prediction: **Defective** {f'— probability {prob:.2%}' if prob is not None else ''}")
            else:
                st.success(f"Prediction: **Non-defective** {f'— probability {1-prob:.2%}' if prob is not None else ''}")

            if warn:
                st.warning(warn)

            # Explain prediction if logistic regression
            if hasattr(model, "coef_"):
                contrib = explain_prediction_logreg(model, scaler, manual_df)
                st.markdown("##### 🔍 Why this prediction?")
                st.caption(f"Per-feature contribution to the decision {contrib.attrs.get('scale_note','')}.")
                top_k = contrib.copy()
                top_k["Direction"] = np.where(top_k["Contribution"]>=0, "Increases defect risk", "Decreases defect risk")
                st.dataframe(top_k, use_container_width=True)

    # Batch predict flow
    if batch_file is not None and model is not None:
        try:
            bdf = pd.read_csv(batch_file)
            yhat, warn = predict_df(model, scaler, bdf)
            if yhat is None:
                st.error(warn)
            else:
                out = bdf.copy()
                out["PredictedStatus"] = np.where(yhat==1, "Defective", "Non-defective")
                st.markdown("#### Results")
                st.dataframe(out.head(50), use_container_width=True)

                # Summary viz
                vc = out["PredictedStatus"].value_counts()
                fig, ax = plt.subplots()
                ax.pie(vc.values, labels=vc.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)

                # Download
                csv_bytes = out.to_csv(index=False).encode('utf-8')
                st.download_button("⬇️ Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
                if warn:
                    st.warning(warn)
        except Exception as e:
            st.error(f"Batch prediction error: {e}")

# ======================= TAB 3: AI ENHANCEMENTS =======================
with tab3:
    st.subheader("AI-Powered Insights & Explanations")
    if ai_ready and GENAI_AVAILABLE:
        try:
            prompt = (
                "You are a manufacturing analytics assistant. Write a clear, concise, executive-level summary for a dashboard "
                "that predicts product defects using Production Volume, Supplier Quality, Maintenance Hours, and Energy Efficiency. "
                "Explain what each feature generally implies about defect risk, what managers should watch for, and 5 actionable recommendations. "
                "Keep it under 200 words in simple language."
            )
            model_gen = genai.GenerativeModel("gemini-pro")
            resp = model_gen.generate_content(prompt)
            st.markdown("#### 📑 AI-Generated Summary")
            st.info(resp.text if hasattr(resp, 'text') else str(resp))
        except Exception as e:
            st.error(f"AI error: {e}")
    else:
        st.warning("Provide a valid Google API key in the sidebar to enable AI summaries.")

# ======================= TAB 4: FINAL INSIGHTS =======================
with tab4:
    st.subheader("Executive Summary Dashboard")

    # If user uploaded data with target, compute quick KPIs using the current model
    if 'df' in locals() and not df.empty:
        target_col = None
        for cand in ["DefectStatus", "Defective", "target", "label"]:
            if cand in df.columns:
                target_col = cand
                break
        if target_col and all(c in df.columns for c in FEATURES):
            # Prepare X, y
            X_eval = df[FEATURES].copy()
            y_true = df[target_col].copy()
            # normalize target to 0/1 if strings
            if y_true.dtype == object:
                y_true = y_true.map({"Defective":1, "Non-defective":0, "Yes":1, "No":0}).fillna(y_true).astype(int)
            try:
                if scaler is not None:
                    y_pred = model.predict(scaler.transform(X_eval))
                    y_proba = model.predict_proba(scaler.transform(X_eval))[:,1]
                else:
                    y_pred = model.predict(X_eval)
                    y_proba = model.predict_proba(X_eval)[:,1]
                acc = float((y_pred==y_true).mean())
                pos_rate = float((y_pred==1).mean())
            except Exception as e:
                acc, pos_rate = None, None
                st.warning(f"Could not score model on provided data: {e}")
        else:
            acc, pos_rate = None, None
    else:
        acc, pos_rate = None, None

    # Metric cards
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown('<div class="metric-card"><h3>Model</h3><div class="val">Logistic Regression</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card"><h3>Accuracy</h3><div class="val">{f"{acc*100:.1f}%" if acc is not None else "—"}</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card"><h3>Predicted Defect Rate</h3><div class="val">{f"{pos_rate*100:.1f}%" if pos_rate is not None else "—"}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### 🔁 Scenario Simulation (What-if)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pv2 = st.number_input("Production Volume", min_value=100, max_value=999, value=500, step=1, key="pv2")
    with c2:
        sq2 = st.number_input("Supplier Quality", min_value=80.0, max_value=100.0, value=90.0, step=0.1, format="%.1f", key="sq2")
    with c3:
        mh2 = st.number_input("Maintenance Hours", min_value=0, max_value=23, value=8, step=1, key="mh2")
    with c4:
        ee2 = st.number_input("Energy Efficiency", min_value=0.1, max_value=0.5, value=0.3, step=0.01, format="%.2f", key="ee2")

    sim_df = pd.DataFrame([[pv2, sq2, mh2, ee2]], columns=FEATURES)
    btn_sim = st.button("Simulate Outcome")

    if btn_sim and model is not None:
        try:
            if scaler is not None:
                proba = float(model.predict_proba(scaler.transform(sim_df))[0,1])
                pred = int(model.predict(scaler.transform(sim_df))[0])
            else:
                proba = float(model.predict_proba(sim_df)[0,1])
                pred = int(model.predict(sim_df)[0])

            if pred == 1:
                st.error(f"Outcome: **Defective** (probability {proba:.1%})")
                if hasattr(model, "coef_"):
                    contrib = explain_prediction_logreg(model, scaler, sim_df)
                    st.markdown("#### Why is it predicted defective?")
                    pos = contrib[contrib["Contribution"]>0].head(3)
                    neg = contrib[contrib["Contribution"]<0].head(3)
                    if not pos.empty:
                        st.write("**Risk-increasing factors:** " + ", ".join(f"{r.Feature} (value {r.Value:.3g})" for r in pos.itertuples()))
                    if not neg.empty:
                        st.write("**Risk-reducing factors:** " + ", ".join(f"{r.Feature} (value {r.Value:.3g})" for r in neg.itertuples()))
                    st.caption(contrib.attrs.get('scale_note',''))
            else:
                st.success(f"Outcome: **Non-defective** (probability {1-proba:.1%})")
        except Exception as e:
            st.error(f"Simulation error: {e}")

    st.markdown("---")

    st.markdown("### 💡 Business Recommendations")
    st.markdown(
        """
- Strengthen **Supplier Quality** audits; prioritize vendors with consistent >95 scores.
- Keep **Maintenance Hours** within planned windows; spikes indicate process instability.
- Track **Energy Efficiency**; values below ~0.30 often correlate with higher defect risk.
- Calibrate **Production Volume** targets to avoid overloading equipment; monitor after big ramps.
- Build a weekly **defect review**: top 3 drivers, actions, and owner accountability.
        """
    )

    # -------- Report download --------
    st.markdown("### 📥 Download Executive Report (PDF)")
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    def build_pdf(summary_text: str) -> bytes:
        buff = io.BytesIO()
        doc = SimpleDocTemplate(buff)
        styles = getSampleStyleSheet()
        story = [
            Paragraph("Manufacturing Defect Prediction – Executive Report", styles['Title']),
            Spacer(1, 12),
            Paragraph("This report summarizes data quality, predictive outcomes, and recommended actions to reduce defects.", styles['BodyText']),
            Spacer(1, 12),
            Paragraph("Key KPIs", styles['Heading2']),
            Paragraph(f"Accuracy: {f'{acc*100:.1f}%' if acc is not None else '—'}", styles['BodyText']),
            Paragraph(f"Predicted Defect Rate: {f'{pos_rate*100:.1f}%' if pos_rate is not None else '—'}", styles['BodyText']),
            Spacer(1, 12),
            Paragraph("Recommendations", styles['Heading2']),
            Paragraph("- Strengthen supplier quality audits and gate incoming material.", styles['BodyText']),
            Paragraph("- Stabilize maintenance schedules to reduce process variability.", styles['BodyText']),
            Paragraph("- Monitor energy efficiency; investigate dips promptly.", styles['BodyText']),
            Paragraph("- Pace production volume increases with equipment readiness.", styles['BodyText']),
        ]
        if summary_text:
            story.extend([Spacer(1, 12), Paragraph("AI Summary", styles['Heading2']), Paragraph(summary_text, styles['BodyText'])])
        doc.build(story)
        return buff.getvalue()

    ai_summary_for_pdf = ""
    if ai_ready and GENAI_AVAILABLE:
        try:
            resp2 = genai.GenerativeModel("gemini-pro").generate_content(
                "Summarize this manufacturing defect dashboard for executives in 80-120 words. Focus on actions.")
            ai_summary_for_pdf = resp2.text if hasattr(resp2, 'text') else str(resp2)
        except Exception:
            ai_summary_for_pdf = ""

    pdf_bytes = build_pdf(ai_summary_for_pdf)
    st.download_button("⬇️ Download PDF", data=pdf_bytes, file_name="Defect_Executive_Report.pdf", mime="application/pdf")
