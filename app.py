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
LABELS = {0: "Survey Candidate", 1: "Confirmed Exoplanet", 2: "False Positive"}

# Add explanatory text for model interpretation
MODEL_EXPLANATION = """
⚠️ **Important**: This model was trained on NASA K2 exoplanet survey data to classify 
astronomical observations, NOT to determine if celestial bodies are "real planets" or not.

**Labels explained:**
- 🔍 **Survey Candidate**: Potential exoplanet signal requiring further verification
- ✅ **Confirmed Exoplanet**: Verified exoplanet from astronomical observations  
- ❌ **False Positive**: Measurement error or stellar activity (not a real planet signal)

**Note**: All solar system planets (Earth, Jupiter, etc.) would classify as "Confirmed Exoplanet" 
in astronomical surveys since they ARE real planets! The model doesn't classify 
"planet vs non-planet" but rather "survey signal quality".
"""


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


def check_data_distribution(X_filled: pd.DataFrame, feature_names: list[str]) -> dict:
	"""Check if input data is within reasonable ranges based on training data expectations."""
	warnings = []
	
	# Define expected ranges for key features (based on exoplanet survey data)
	expected_ranges = {
		'pl_orbper': (0.5, 10000),  # orbital period in days
		'pl_rade': (0.1, 30),       # planet radius in Earth radii  
		'st_rad': (0.1, 50),        # stellar radius in solar radii
		'st_mass': (0.1, 10),       # stellar mass in solar masses
		'pl_insol': (0.001, 1000),  # insolation flux
		'st_teff': (2000, 10000),   # stellar temperature in K
	}
	
	for feature in feature_names:
		if feature in expected_ranges and feature in X_filled.columns:
			values = X_filled[feature].dropna()
			if len(values) > 0:
				min_val, max_val = expected_ranges[feature]
				out_of_range = values[(values < min_val) | (values > max_val)]
				if len(out_of_range) > 0:
					warnings.append(f"⚠️ **{feature}**: {len(out_of_range)} values outside expected range [{min_val}, {max_val}]")
	
	return {"warnings": warnings, "is_reliable": len(warnings) == 0}


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
	
	# Debug path issues
	import os
	st.write(f"🔍 Current working directory: {os.getcwd()}")
	st.write(f"🔍 Model path: {MODEL_PATH}")
	st.write(f"🔍 Model path absolute: {MODEL_PATH.absolute()}")
	st.write(f"🔍 Model exists: {MODEL_PATH.exists()}")
	
	if not MODEL_PATH.exists():
		st.error("Model file not found. Please train the model first.")
		st.stop()

	sample_toggle = st.toggle("Use sample NASA K2 CSV", value=True)
	uploaded = st.file_uploader("Upload a CSV (NASA format ok)", type=["csv"]) if not sample_toggle else None
	row_select = st.number_input("Row index for explanation", min_value=0, value=0, step=1)


# ----------------------------
# Header
# ----------------------------
st.markdown("""
# 🌌 Exoplanet Survey Classification (K2)
Classify astronomical survey signals using NASA K2 mission data patterns.
""")

# Add model explanation
with st.expander("📖 How to interpret predictions", expanded=False):
	st.markdown(MODEL_EXPLANATION)

# Add warning about solar system data
st.info("💡 **Testing with Solar System planets?** They will correctly classify as 'Confirmed Exoplanet' since they ARE real planets detected in surveys!")


# ----------------------------
# Load model/pipeline
# ----------------------------
st.write(f"🤖 Loading model from: {MODEL_PATH}")
if not MODEL_PATH.exists():
	st.error(f"❌ Model file not found at: {MODEL_PATH}")
	st.stop()

try:
	pipe, model, imputer, scaler, feature_names, bias = load_pipeline(MODEL_PATH)
	explainer = get_explainer(model)
	st.success(f"✅ Model loaded successfully! Features: {len(feature_names)}")
except Exception as e:
	st.error(f"❌ Error loading model: {e}")
	st.stop()


# ----------------------------
# Data acquisition
# ----------------------------
if sample_toggle:
	# Use sample K2 PANDC file from data
	data_files = sorted(Path("data").glob("k2pandc_*.csv"))
	st.write(f"🔍 Debug: Found {len(data_files)} data files: {[f.name for f in data_files]}")
	if data_files:
		selected_file = data_files[-1]
		st.write(f"📂 Loading file: {selected_file}")
		try:
			df_raw = pd.read_csv(selected_file, comment="#")
			st.success(f"✅ Successfully loaded {len(df_raw)} rows from {selected_file.name}")
		except Exception as e:
			st.error(f"❌ Error loading file: {e}")
			df_raw = None
	else:
		st.warning("No sample K2 CSV found in data/. Please upload a file.")
		df_raw = None
else:
	if uploaded is not None:
		try:
			df_raw = pd.read_csv(uploaded, comment="#")
			st.success(f"✅ Successfully loaded {len(df_raw)} rows from uploaded file")
		except Exception as e:
			st.error(f"❌ Error loading uploaded file: {e}")
			df_raw = None
	else:
		df_raw = None

if df_raw is None:
	st.stop()

st.markdown("### 🔎 Sample of Input Data")
st.dataframe(df_raw.head(10), use_container_width=True)


# ----------------------------
# Preprocess and Predict
# ----------------------------
Xa, X_df, coverage = prepare_features(df_raw, feature_names, imputer=imputer, scaler=scaler)

# Check data distribution and show warnings
distribution_check = check_data_distribution(X_df, feature_names)
if not distribution_check["is_reliable"]:
	with st.expander("⚠️ Data Quality Warnings", expanded=True):
		st.warning("**Input data may be outside the model's training distribution:**")
		for warning in distribution_check["warnings"]:
			st.markdown(f"- {warning}")
		st.markdown("**This may lead to unreliable predictions.** Model was trained on exoplanet survey data with specific ranges.")

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
		use_container_width=True
	)

with col2:
	st.plotly_chart(plot_probabilities(proba, row_select), use_container_width=True)


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
		use_container_width=True
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
st.plotly_chart(fig_counts, use_container_width=True)

st.caption("Data source: NASA Exoplanet Archive • Model: XGBoost (K2-adapted) • Explainability: SHAP")

