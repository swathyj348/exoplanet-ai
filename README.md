Exoplanet AI - Minimal Pipeline

Scripts
- src/preprocess.py: Build merged_tabular.csv from Kepler/K2 (handles NASA headers, maps labels).
- src/train_k2_adapt.py: Train optimized XGBoost adapted to K2; saves models/xgb_k2_adapt.pkl.
- src/evaluate.py: Evaluate any saved pipeline on a CSV; supports saved imputer/scaler/bias.

Quick start
1) Train (K2-adapted):
   - python src/train_k2_adapt.py
2) Evaluate on K2 PANDC:
   - python src/evaluate.py --model models/xgb_k2_adapt.pkl --csv data/k2pandc_2025.10.04_07.36.56.csv --out reports

Notes
- The saved pipeline includes a median SimpleImputer, StandardScaler, and a small per-class probability bias tuned on validation. You can override bias with --bias.