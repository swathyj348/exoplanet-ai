import io
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st


# ----------------------------
# App Config & Styling
# ----------------------------
st.set_page_config(
	page_title="Exoplanet Classifier (K2)",
	page_icon="🪐",
	layout="wide",
)

NASA_BG = """
<style>
/* Dark, NASA-inspired theme */
html, body, [class*="css"], .stApp {
  background: radial-gradient(ellipse at top, #0b0f1a 0%, #02040a 60%), radial-gradient(ellipse at bottom, #0b0f1a 0%, #02040a 60%);
  color: #E6EDF3;
}

.block-container {padding-top: 2rem; padding-bottom: 2rem;}
/* Card look */
.card {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; padding: 1rem;}
.accent {color: #8BB9FF;}
.success {color: #00D3A7;}
.warn {color: #FFB86C;}

/* Buttons */
div.stButton>button {background: linear-gradient(90deg, #223654, #11233b); color: #E6EDF3; border: 1px solid #2d415f;}
div.stDownloadButton>button {background: linear-gradient(90deg, #2a3b5c, #172338); color: #E6EDF3; border: 1px solid #344760;}

/* Headers */
h1,h2,h3 { color: #E6EDF3; }

/* Tables */
.stDataFrame, .stTable { background: rgba(255,255,255,0.03) }
</style>
"""
st.markdown(NASA_BG, unsafe_allow_html=True)


# ----------------------------
# Utilities
# ----------------------------
MODEL_PATH = Path("models/xgb_k2_adapt.pkl")
LABELS = {0: "Candidate", 1: "Confirmed Exoplanet", 2: "False Positive"}


@st.cache_resource(show_spinner=False)
def load_pipeline(path: Path):
	with open(path, "rb") as f:
		pipe = pickle.load(f)
	model = pipe["model"]
	scaler = pipe.get("scaler")
	imputer = pipe.get("imputer")
	bias = pipe.get("bias")
	feature_names = pipe["feature_names"]
	return pipe, model, imputer, scaler, feature_names, bias


def prepare_features(df: pd.DataFrame, feature_names: list[str], imputer=None, scaler=None):
	# normalize names
	df = df.copy()
	df.columns = df.columns.str.lower().str.strip()

	# build feature frame
	X = pd.DataFrame(index=df.index, columns=feature_names)
	present = [c for c in feature_names if c in df.columns]
	missing = [c for c in feature_names if c not in df.columns]
	if present:
		X.loc[:, present] = df[present].apply(pd.to_numeric, errors="coerce")
	if missing:
		X.loc[:, missing] = np.nan

	# Convert to array first, then use the trained imputer (not CSV medians!)
	Xa = X.astype(np.float32).to_numpy()
	if imputer is not None:
		# Use the training-time imputer which has the correct medians
		Xa = imputer.transform(Xa)
	else:
		# Fallback: compute medians from the current data (problematic for small uploads)
		med = np.nanmedian(Xa, axis=0)
		inds = np.where(np.isnan(Xa))
		Xa[inds] = np.take(med, inds[1])
	
	# Convert back to DataFrame for log1p transform
	X_filled = pd.DataFrame(Xa, columns=feature_names, index=X.index)
	
	# Apply the same log1p transform used in training for skewed columns
	skew_cols = [c for c in ['pl_orbper', 'koi_period', 'pl_rade', 'koi_prad', 'st_rad', 'st_mass', 'pl_insol'] if c in X_filled.columns]
	if skew_cols:
		skew_mat = X_filled[skew_cols].to_numpy(dtype=np.float64, copy=True)
		np.clip(skew_mat, a_min=0, a_max=None, out=skew_mat)
		skew_mat = np.log1p(skew_mat)
		X_filled.loc[:, skew_cols] = skew_mat

	Xa = X_filled.astype(np.float32).to_numpy()
	if scaler is not None:
		Xa = scaler.transform(Xa)
	coverage = {
		'present': len(present),
		'missing': len(missing),
		'total': len(feature_names),
		'present_cols': present,
		'missing_cols': missing,
	}
	return Xa, X, coverage


def apply_bias(proba: np.ndarray, bias: list[float] | np.ndarray | None) -> np.ndarray:
	if bias is None:
		return proba
	b = np.array(bias, dtype=float)
	if b.shape[0] != proba.shape[1]:
		return proba
	p = proba * b[None, :]
	p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
	return p


@st.cache_resource(show_spinner=False)
def get_explainer(_model):
	try:
		# Works well for tree models (XGBoost)
		explainer = shap.TreeExplainer(_model)
	except Exception:
		explainer = shap.Explainer(_model)
	return explainer


def compute_shap_for_row(explainer, X_row: np.ndarray, predicted_class: int):
	# Compute SHAP only for one row to keep it fast
	# New SHAP API returns Explanation; old returns arrays. Try both.
	try:
		explanation = explainer(X_row)
		# Explanation.values shape: (1, n_features, n_classes) for multiclass
		vals = np.array(explanation.values)
		base = np.array(explanation.base_values)
		if vals.ndim == 3:  # multiclass
			vals_c = vals[0, :, predicted_class]
			base_c = base[0, predicted_class] if base.ndim == 2 else base[predicted_class]
		else:  # binary/regression style
			vals_c = vals[0]
			base_c = base[0] if np.ndim(base) else float(base)
		return vals_c, base_c
	except Exception:
		# Legacy path
		shap_values = explainer.shap_values(X_row)
		if isinstance(shap_values, list):
			vals_c = shap_values[predicted_class][0]
		else:
			vals_c = shap_values[0]
		base_c = explainer.expected_value[predicted_class] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
		return np.array(vals_c), float(base_c)


def plot_probabilities(proba: np.ndarray, idx: int):
	vals = proba[idx]
	labels = [LABELS[i] for i in range(len(vals))]
	fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=["#7aa1f7", "#00d3a7", "#ffb86c"]))
	fig.update_layout(template="plotly_dark", yaxis_title="Probability", xaxis_title="Class",
					  title="Prediction Probabilities", height=350)
	return fig


def make_waterfall(feature_names: list[str], shap_values_row: np.ndarray, base_value: float, proba: np.ndarray, cls_idx: int):
	# Simple waterfall using Plotly to avoid heavy matplotlib in Streamlit
	# Sort features by absolute contribution
	order = np.argsort(np.abs(shap_values_row))[::-1]
	names = [feature_names[i] for i in order]
	contribs = shap_values_row[order]

	cum = base_value
	x = []
	y = []
	measure = []
	for name, c in zip(names[:15], contribs[:15]):  # top-15
		x.append(name)
		y.append(c)
		measure.append("relative")

	fig = go.Figure(go.Waterfall(
		name="SHAP",
		orientation="v",
		measure=measure,
		x=x,
		text=[f"{v:+.3f}" for v in y],
		y=y,
		connector={"line": {"color": "rgba(200,200,200,0.2)"}},
	))
	fig.update_layout(template="plotly_dark", title=f"Feature Contributions → Class: {LABELS[cls_idx]}",
					  yaxis_title="SHAP value", height=420)
	return fig


# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("🪐 Exoplanet Classifier")
st.sidebar.caption("K2-adapted XGBoost • NASA-inspired UI")

with st.sidebar:
	st.markdown("**Model:** models/xgb_k2_adapt.pkl")
	if not MODEL_PATH.exists():
		st.error("Model file not found. Please train the model first.")

	sample_toggle = st.toggle("Use sample NASA K2 CSV", value=True)
	uploaded = st.file_uploader("Upload a CSV (NASA format ok)", type=["csv"]) if not sample_toggle else None
	row_select = st.number_input("Row index for explanation", min_value=0, value=0, step=1)


# ----------------------------
# Header
# ----------------------------
st.markdown("""
# 🌌 Exoplanet Classification (K2)
Predict whether an object is a Candidate, Confirmed Exoplanet, or a False Positive, with explainability.
""")


# ----------------------------
# Load model/pipeline
# ----------------------------
pipe, model, imputer, scaler, feature_names, bias = load_pipeline(MODEL_PATH)
explainer = get_explainer(model)


# ----------------------------
# Data acquisition
# ----------------------------
if sample_toggle:
	# Use sample K2 PANDC file from data
	data_files = sorted(Path("data").glob("k2pandc_*.csv"))
	if data_files:
		df_raw = pd.read_csv(data_files[-1], comment="#")
	else:
		st.warning("No sample K2 CSV found in data/. Please upload a file.")
		df_raw = None
else:
	if uploaded is not None:
		df_raw = pd.read_csv(uploaded, comment="#")
	else:
		df_raw = None

if df_raw is None:
	st.stop()

st.markdown("### 🔎 Sample of Input Data")
st.dataframe(df_raw.head(10), width='stretch')


# ----------------------------
# Preprocess and Predict
# ----------------------------
Xa, X_df, coverage = prepare_features(df_raw, feature_names, imputer=imputer, scaler=scaler)
if coverage['missing'] > 0:
	miss_ratio = coverage['missing'] / max(coverage['total'], 1)
	if miss_ratio >= 0.5:
		st.warning(f"Only {coverage['present']} of {coverage['total']} expected features present (\n"
				   f"{miss_ratio:.0%} missing). Predictions may be uniform or inaccurate.")
	else:
		st.info(f"Using {coverage['present']} of {coverage['total']} expected features. Missing: {coverage['missing']}")
proba = model.predict_proba(Xa)
proba = apply_bias(proba, bias)
preds = np.argmax(proba, axis=1)

# Save probabilities to CSV (in-memory)
proba_df = pd.DataFrame(
	proba, columns=[f"prob_{LABELS[i].replace(' ', '_').lower()}" for i in range(proba.shape[1])]
)
out_df = pd.concat([X_df.reset_index(drop=True), pd.Series(preds, name="prediction"), proba_df], axis=1)
csv_buf = io.BytesIO()
out_df.to_csv(csv_buf, index=False)
csv_buf.seek(0)


# ----------------------------
# Results: Prediction + Probabilities
# ----------------------------
st.markdown("## 🚀 Prediction Results")

if row_select >= len(preds):
	st.warning(f"Row index {row_select} out of range; showing row 0 instead.")
	row_select = 0

col1, col2 = st.columns([1, 1])
with col1:
	st.markdown("### Top-level Prediction")
	st.metric(
		label="Predicted Class",
		value=f"{LABELS[int(preds[row_select])]}",
		delta=f"p={proba[row_select, int(preds[row_select])]:.3f}",
		help="Probability of the predicted class for the selected row",
	)
	st.download_button(
		label="📥 Download Probabilities CSV",
		data=csv_buf,
		file_name="exoplanet_predictions.csv",
		mime="text/csv",
		width='stretch',
	)

with col2:
	st.plotly_chart(plot_probabilities(proba, row_select), width='stretch', theme="streamlit")


# ----------------------------
# Explainability (SHAP) — directly below predictions
# ----------------------------
st.markdown("## 🛰️ Explainability (SHAP)")
try:
	x_row = Xa[row_select : row_select + 1]
	cls_idx = int(preds[row_select])
	shap_vals_row, base_val = compute_shap_for_row(explainer, x_row, cls_idx)
	st.plotly_chart(
		make_waterfall(feature_names, shap_vals_row, base_val, proba, cls_idx),
		width='stretch',
		theme="streamlit",
	)
except Exception as e:
	st.warning(f"SHAP explainability unavailable: {e}")


# ----------------------------
# Extra visuals (optional)
# ----------------------------
st.markdown("### 📊 Distribution of Predicted Classes")
cls_counts = pd.Series(preds).map(LABELS).value_counts()
fig_counts = px.pie(values=cls_counts.values, names=cls_counts.index, hole=0.45, template="plotly_dark")
fig_counts.update_layout(height=360)
st.plotly_chart(fig_counts, width='stretch', theme="streamlit")

st.caption("Data source: NASA Exoplanet Archive • Model: XGBoost (K2-adapted) • Explainability: SHAP")

