"""
Tabular machine learning training module for exoplanet classification.

This module trains XGBoost classifier on preprocessed tabular features
and evaluates their performance.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import pickle
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_xgboost_classifier():
    """
    Main training function that loads data, trains XGBoost classifier, and saves pipeline.
    """
    print("Starting XGBoost training...")
    
    # 1. Load datasets
    print("Loading datasets...")
    merged = pd.read_csv('data/merged_tabular.csv')
    print(f"Loaded merged tabular data: {merged.shape}")
    
    # Try to load and merge time series features (optional)
    try:
        ts = pd.read_csv('data/ts_features.csv')
        print(f"Loaded TS features: {ts.shape}")
        
        # Merge on common ID column (if exists)
        if 'id' in merged.columns and 'id' in ts.columns:
            merged = pd.merge(merged, ts, left_on='id', right_on='id', how='left')
            print(f"Merged with TS features: {merged.shape}")
        else:
            print("No common ID column found, continuing with tabular only")
    except FileNotFoundError:
        print('TS features not found, continuing with tabular only')
    except Exception as e:
        print(f'Error loading TS features: {e}, continuing with tabular only')
    
    # 2. Split features/labels
    print("Preparing features and labels...")
    X = merged.drop(columns=['label', 'id'], errors='ignore')
    y = merged['label']
    
    print(f"Features shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    print(f"Label distribution:\n{y.value_counts()}")
    
    # Handle missing values in features
    X = X.fillna(X.median())
    
    # 3. Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Features scaled successfully")
    
    # 4. Train/test split
    print("Splitting data...")
    Xtr, Xte, ytr, yte = train_test_split(
        X_scaled, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=42
    )
    
    print(f"Training set: {Xtr.shape[0]} samples")
    print(f"Test set: {Xte.shape[0]} samples")
    print(f"Training label distribution:\n{pd.Series(ytr).value_counts()}")
    
    # 5. Compute class imbalance weights
    print("Computing class weights...")
    
    # For multi-class (3 classes: 0=CANDIDATE, 1=CONFIRMED, 2=FALSE POSITIVE)
    # Calculate sample weights for each class
    unique_classes = np.unique(ytr)
    n_samples = len(ytr)
    n_classes = len(unique_classes)
    
    # Compute class weights
    class_weights = {}
    for cls in unique_classes:
        class_weights[cls] = n_samples / (n_classes * np.sum(ytr == cls))
    
    print(f"Class weights: {class_weights}")
    
    # For XGBoost, we'll use scale_pos_weight for binary case or sample_weight for multi-class
    if len(unique_classes) == 2:
        # Binary classification
        pos_class = 1
        neg_class = 0
        scale_pos_weight = np.sum(ytr == neg_class) / np.sum(ytr == pos_class)
        print(f"Scale pos weight: {scale_pos_weight}")
        
        # 6. Train XGBoost (binary)
        print("Training XGBoost classifier (binary)...")
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
    else:
        # Multi-class classification
        print("Training XGBoost classifier (multi-class)...")
        
        # Create sample weights for training
        sample_weights = np.array([class_weights[cls] for cls in ytr])
        
        model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        # Train with sample weights
        model.fit(
            Xtr, ytr,
            sample_weight=sample_weights,
            eval_set=[(Xte, yte)],
            verbose=True
        )
    
    # If binary, train without sample weights but with scale_pos_weight
    if len(unique_classes) == 2:
        model.fit(
            Xtr, ytr,
            eval_set=[(Xte, yte)],
            verbose=True
        )
    
    # Evaluate on test set
    print("\nEvaluating model...")
    yte_pred = model.predict(Xte)
    yte_proba = model.predict_proba(Xte)
    
    accuracy = accuracy_score(yte, yte_pred)
    f1 = f1_score(yte, yte_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(yte, yte_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(yte, yte_pred))
    
    # 7. Save pipeline
    print("Saving model pipeline...")
    
    # Create models directory if it doesn't exist
    import os
    os.makedirs('models', exist_ok=True)
    
    pipeline = {
        'model': model,
        'scaler': scaler,
        'feature_names': list(X.columns),
        'class_weights': class_weights if len(unique_classes) > 2 else {'scale_pos_weight': scale_pos_weight},
        'metrics': {
            'test_accuracy': accuracy,
            'test_f1': f1
        }
    }
    
    with open('models/xgb_tabular.pkl', 'wb') as f:
        pickle.dump(pipeline, f)
    
    print('Saved xgb_tabular.pkl')
    
    return model, scaler, X.columns, Xtr, yte, yte_pred


if __name__ == "__main__":
    # Run the training
    model, scaler, feature_names, Xtr, yte, yte_pred = train_xgboost_classifier()
    
    print("\nTraining completed successfully!")
    print("Model saved to: models/xgb_tabular.pkl")
    print("Ready for evaluation with evaluate.py")
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)