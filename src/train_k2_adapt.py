"""
Train an XGBoost classifier adapted to K2 PANDC data.

Steps:
- Load K2 CSV (NASA Archive comment headers)
- Map disposition to labels {CANDIDATE:0, CONFIRMED:1, FALSE POSITIVE:2}
- Select robust numeric features available in K2, align columns
- Split into train/val/test with stratification
- Small hyperparameter sweep on validation accuracy
- Train best model on train+val, evaluate on test
- Save pipeline to models/xgb_k2_adapt.pkl
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier


LABEL_MAP = {
    'CONFIRMED': 1, 'confirmed': 1,
    'CANDIDATE': 0, 'candidate': 0,
    'FALSE POSITIVE': 2, 'false positive': 2,
    'PC': 0,  # Planet Candidate
    'CP': 1,  # Confirmed Planet
    'FP': 2,  # False Positive
    'KP': 1,  # Known Planet
}


POTENTIAL_NUMERIC_COLS = [
    # Orbital / period
    'pl_orbper', 'koi_period', 'pl_orbsmax',
    # Planet radius / mass proxies
    'pl_rade', 'koi_prad', 'pl_bmasse', 'pl_bmassj',
    # Eccentricity
    'pl_orbeccen', 'koi_ecc',
    # Stellar properties
    'st_teff', 'koi_steff', 'st_rad', 'koi_srad', 'st_mass', 'koi_smass',
    # Magnitude
    'sy_kepmag', 'koi_kepmag',
    # Ratios and derived
    'pl_ratdor', 'koi_dor', 'pl_insol', 'pl_eqt',
    # System context
    'sy_snum', 'sy_pnum', 'disc_year'
]


def load_k2(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, comment='#')
    df.columns = df.columns.str.lower().str.strip()
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    target_col = None
    for cand in ['koi_disposition', 'disposition', 'tfopwg_disp']:
        if cand in df.columns:
            target_col = cand
            break
    if not target_col:
        raise ValueError('No disposition column found in K2 CSV')

    y = df[target_col].astype(str).str.upper().map(LABEL_MAP)
    mask = y.notna()
    df = df.loc[mask].copy()
    df['label'] = y.loc[mask].astype(int)
    df.drop(columns=[target_col], inplace=True)
    return df


def select_features(df: pd.DataFrame) -> List[str]:
    available = [c for c in POTENTIAL_NUMERIC_COLS if c in df.columns]
    # Ensure at least a minimal set
    if len(available) < 6:
        # Fallback: pick first 10 numeric columns present
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        available = [c for c in num_cols if c != 'label'][:10]
    return available


def build_matrices(df: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    # Build numeric frame with required features
    X = pd.DataFrame(index=df.index, columns=features)
    present = [c for c in features if c in df.columns]
    missing = [c for c in features if c not in df.columns]
    if present:
        X.loc[:, present] = df[present].apply(pd.to_numeric, errors='coerce')
    if missing:
        X.loc[:, missing] = np.nan

    # Median imputation in one pass
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    # Fallback all-NaN to zeros
    all_nan_cols = [c for c in features if X[c].isna().all()]
    if all_nan_cols:
        X.loc[:, all_nan_cols] = 0.0

    # Apply log1p to positively skewed features in bulk
    skew_cols = [c for c in ['pl_orbper', 'koi_period', 'pl_rade', 'koi_prad', 'st_rad', 'st_mass', 'pl_insol'] if c in X.columns]
    if skew_cols:
        skew_mat = X[skew_cols].to_numpy(dtype=np.float64, copy=True)
        np.clip(skew_mat, a_min=0, a_max=None, out=skew_mat)
        skew_mat = np.log1p(skew_mat)
        X.loc[:, skew_cols] = skew_mat

    # Cast to float32 for efficiency
    X = X.astype('float32')

    y = df['label'].to_numpy()
    return X.to_numpy(), y, X


def small_param_grid():
    return [
        # shallower-fast
        dict(n_estimators=500, max_depth=4, learning_rate=0.12, subsample=1.0, colsample_bytree=1.0, min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0),
        dict(n_estimators=800, max_depth=5, learning_rate=0.08, subsample=0.9, colsample_bytree=0.9, min_child_weight=2, reg_lambda=1.0, reg_alpha=0.0),
        # baseline
        dict(n_estimators=700, max_depth=6, learning_rate=0.08, subsample=1.0, colsample_bytree=0.9, min_child_weight=1, reg_lambda=1.0, reg_alpha=0.0),
        dict(n_estimators=900, max_depth=6, learning_rate=0.06, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, reg_lambda=1.0, reg_alpha=0.0),
        # deeper + regularized
        dict(n_estimators=1000, max_depth=7, learning_rate=0.05, subsample=0.9, colsample_bytree=0.8, min_child_weight=4, reg_lambda=2.0, reg_alpha=0.5),
        dict(n_estimators=1200, max_depth=8, learning_rate=0.04, subsample=0.85, colsample_bytree=0.8, min_child_weight=6, reg_lambda=3.0, reg_alpha=1.0),
    ]


def main():
    csv_path = 'data/k2pandc_2025.10.04_07.36.56.csv'
    print('Loading K2 data...')
    df = load_k2(csv_path)
    print(f'K2 raw shape: {df.shape}')

    print('Mapping labels...')
    df = map_labels(df)
    print(f'Labeled K2 shape: {df.shape}')
    print(df['label'].value_counts())

    print('Selecting features...')
    features = select_features(df)
    print(f'Using features: {features}')

    X_all, y_all, X_df = build_matrices(df, features)

    # Split train/val/test: 70/15/15
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_all, y_all, test_size=0.30, stratify=y_all, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )
    print(f'Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}')

    # Persisted imputer (median) and scaler pipeline pieces
    imputer = SimpleImputer(strategy='median')
    X_train_imp = imputer.fit_transform(X_train.astype('float32'))
    X_val_imp = imputer.transform(X_val.astype('float32'))
    X_test_imp = imputer.transform(X_test.astype('float32'))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train_imp)
    X_val_s = scaler.transform(X_val_imp)
    X_test_s = scaler.transform(X_test_imp)

    # Class weights via sample_weight
    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {cls: (len(y_train) / (len(classes) * cnt)) for cls, cnt in zip(classes, counts)}
    sw_train = np.array([class_weight[c] for c in y_train])

    print('Hyperparameter sweep...')
    best = None
    for i, params in enumerate(small_param_grid(), start=1):
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_jobs=-1,
            random_state=42,
            **params,
        )
        model.fit(X_train_s, y_train, sample_weight=sw_train, eval_set=[(X_val_s, y_val)], verbose=False)
        pred_val = model.predict(X_val_s)
        acc = accuracy_score(y_val, pred_val)
        print(f'[{i}] params={params} -> val_acc={acc:.4f}')
        if not best or acc > best['val_acc']:
            best = {'model': model, 'params': params, 'val_acc': acc}

    print(f"Best val accuracy: {best['val_acc']:.4f} with params: {best['params']}")

    # Retrain on train+val with best params
    X_trv = np.vstack([X_train_s, X_val_s])
    y_trv = np.concatenate([y_train, y_val])
    sw_trv = np.array([class_weight[c] for c in y_trv])
    final_model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        n_jobs=-1,
        random_state=42,
        **best['params'],
    )
    final_model.fit(X_trv, y_trv, sample_weight=sw_trv, eval_set=[(X_test_s, y_test)], verbose=False)

    # Evaluate
    # Evaluate with optional probability bias sweep for best accuracy
    proba_val = final_model.predict_proba(X_val_s)
    proba_test = final_model.predict_proba(X_test_s)

    def apply_bias(proba: np.ndarray, bias_vec: np.ndarray) -> np.ndarray:
        p = proba * bias_vec[None, :]
        p = p / np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        return p

    candidate_biases = [
        np.array([1.0, 1.0, 1.0], dtype=np.float32),
        np.array([1.2, 1.0, 1.0], dtype=np.float32),
        np.array([1.0, 1.0, 1.2], dtype=np.float32),
        np.array([1.2, 1.0, 1.2], dtype=np.float32),
        np.array([1.1, 0.95, 1.15], dtype=np.float32),
    ]
    best_bias = candidate_biases[0]
    best_bias_acc = 0.0
    for b in candidate_biases:
        pred_val_b = np.argmax(apply_bias(proba_val, b), axis=1)
        acc_b = accuracy_score(y_val, pred_val_b)
        if acc_b > best_bias_acc:
            best_bias_acc = acc_b
            best_bias = b

    y_pred = np.argmax(apply_bias(proba_test, best_bias), axis=1)
    test_acc = accuracy_score(y_test, y_pred)
    print(f'Test Accuracy (K2): {test_acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))

    # Save pipeline
    os.makedirs('models', exist_ok=True)
    pipeline = {
        'model': final_model,
        'imputer': imputer,
        'scaler': scaler,
        'feature_names': features,
        'bias': best_bias.tolist(),
        'metrics': {
            'val_accuracy': float(best['val_acc']),
            'test_accuracy': float(test_acc),
        }
    }
    out_path = 'models/xgb_k2_adapt.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f'Saved {out_path}')

    # Write a small report
    os.makedirs('reports', exist_ok=True)
    with open('reports/k2_train_report.json', 'w') as f:
        json.dump(pipeline['metrics'], f, indent=2)
    print('Saved reports/k2_train_report.json')


if __name__ == '__main__':
    main()
