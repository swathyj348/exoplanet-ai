"""
Evaluation script for the exoplanet XGBoost tabular model.

Loads the trained pipeline (scaler + model) and evaluates on a provided CSV.
Computes accuracy, precision (macro/weighted), classification report, and confusion matrix.
"""

import os
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, classification_report, confusion_matrix


LABEL_MAP = {
    'CONFIRMED': 1, 'confirmed': 1,
    'CANDIDATE': 0, 'candidate': 0,
    'FALSE POSITIVE': 2, 'false positive': 2,
    'PC': 0,  # Planet Candidate
    'CP': 1,  # Confirmed Planet
    'FP': 2,  # False Positive
    'KP': 1,  # Known Planet
}


def load_pipeline(model_path: str):
    with open(model_path, 'rb') as f:
        pipeline = pickle.load(f)
    model = pipeline['model']
    scaler = pipeline.get('scaler')
    imputer = pipeline.get('imputer')
    saved_bias = pipeline.get('bias')
    feature_names = pipeline['feature_names']
    return model, scaler, feature_names, saved_bias, pipeline


def prepare_features_from_csv(csv_path: str, feature_names: list[str]):
    # NASA Archive CSVs include comment headers, use comment="#"
    df = pd.read_csv(csv_path, comment='#')
    # normalize column names
    df.columns = df.columns.str.lower().str.strip()

    # map disposition/labels if present
    y_true = None
    target_col = None
    for cand in ['koi_disposition', 'disposition', 'tfopwg_disp']:
        if cand in df.columns:
            target_col = cand
            break
    if target_col:
        y_true = df[target_col].astype(str).str.upper().map(LABEL_MAP)
        # drop rows with unmapped labels
        mask = y_true.notna()
        if not mask.all():
            df = df.loc[mask]
            y_true = y_true.loc[mask]
        y_true = y_true.astype(int).to_numpy()

    # Build matrix with required features only
    cols_present = [c for c in feature_names if c in df.columns]
    cols_missing = [c for c in feature_names if c not in df.columns]

    X = pd.DataFrame(index=df.index, columns=feature_names)
    if cols_present:
        X.loc[:, cols_present] = df[cols_present].apply(pd.to_numeric, errors='coerce')
    if cols_missing:
        X.loc[:, cols_missing] = np.nan

    # Compute medians once and fill
    med = X.median(numeric_only=True)
    X = X.fillna(med)

    # Fallback for all-NaN columns: fill zeros
    all_nan_cols = [c for c in feature_names if X[c].isna().all()]
    if all_nan_cols:
        X.loc[:, all_nan_cols] = 0.0

    # Cast to float32 for efficiency
    X = X.astype('float32')
    return X, y_true


def evaluate_on_file(model_path: str, csv_path: str, bias: list[float] | None = None, out_dir: str | None = None):
    model, scaler, feature_names, saved_bias, pipeline = load_pipeline(model_path)

    # Prepare features
    X, y_true = prepare_features_from_csv(csv_path, feature_names)

    # Use persisted imputer/scaler if present
    X_arr = X.to_numpy(dtype=np.float32)
    if pipeline.get('imputer') is not None:
        X_arr = pipeline['imputer'].transform(X_arr)
    if pipeline.get('scaler') is not None:
        X_arr = pipeline['scaler'].transform(X_arr)
    X_scaled = X_arr

    # Predict probabilities and optionally reweight by bias
    proba = None
    y_pred = None
    try:
        proba = model.predict_proba(X_scaled)
        # Prefer CLI bias; otherwise use saved pipeline bias if available
        eff_bias = bias if bias is not None else saved_bias
        if eff_bias is not None:
            bias_arr = np.array(eff_bias, dtype=float)
            if proba.shape[1] == bias_arr.shape[0]:
                proba = proba * bias_arr
                proba = proba / np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
        y_pred = np.argmax(proba, axis=1)
    except Exception:
        # Fallback to direct predict
        y_pred = model.predict(X_scaled)

    results = {}
    if y_true is not None:
        results['accuracy'] = float(accuracy_score(y_true, y_pred))
        results['precision_macro'] = float(precision_score(y_true, y_pred, average='macro', zero_division=0))
        results['precision_weighted'] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))

        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision (macro): {results['precision_macro']:.4f}")
        print(f"Precision (weighted): {results['precision_weighted']:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

        print("Confusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
    else:
        print("No ground-truth label column found in the CSV; computed predictions only.")

    # Optionally write results to out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        metrics_path = os.path.join(out_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained XGBoost model on a CSV file.')
    parser.add_argument('--model', default='models/xgb_tabular.pkl', help='Path to saved pipeline pickle')
    parser.add_argument('--csv', default='data/k2pandc_2025.10.04_07.36.56.csv', help='Path to evaluation CSV')
    parser.add_argument('--out', default=None, help='Directory to save metrics/report')
    parser.add_argument('--bias', default=None, help='Comma-separated bias multipliers per class, e.g., 2.0,1.0,0.8')
    args = parser.parse_args()

    bias = None
    if args.bias:
        try:
            bias = [float(x) for x in args.bias.split(',')]
        except Exception:
            print('Invalid --bias format; expected comma-separated floats like 2.0,1.0,0.8')
            bias = None

    evaluate_on_file(args.model, args.csv, bias=bias, out_dir=args.out)


if __name__ == '__main__':
    main()
"""
Model evaluation module for exoplanet classification.

This module provides comprehensive evaluation metrics, visualizations,
and model comparison tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import learning_curve, validation_curve
import shap
from typing import Dict, List, Tuple, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Comprehensive model evaluation and comparison tool.
    """
    
    def __init__(self):
        """Initialize the model evaluator."""
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None, average='weighted'):
        """
        Calculate comprehensive classification metrics.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            y_pred_proba (array): Predicted probabilities
            average (str): Averaging method for multi-class metrics
            
        Returns:
            dict: Dictionary of metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average=average, zero_division=0)
        }
        
        # AUC score if probabilities are provided
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_true)) == 2:
                    # Binary classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                else:
                    # Multi-class classification
                    metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate AUC: {str(e)}")
                metrics['auc_roc'] = None
        
        return metrics
    
    def evaluate_single_model(self, model, X_test, y_test, model_name="Model", 
                             feature_names=None):
        """
        Comprehensive evaluation of a single model.
        
        Args:
            model: Trained model
            X_test (array): Test features
            y_test (array): Test labels
            model_name (str): Name of the model
            feature_names (list): Names of features for interpretability
            
        Returns:
            dict: Evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        
        try:
            y_pred_proba = model.predict_proba(X_test)
        except AttributeError:
            y_pred_proba = None
            logger.warning(f"{model_name} does not support probability prediction")
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Store results
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': class_report
        }
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            results['feature_importance'] = model.feature_importances_
            if feature_names:
                results['feature_names'] = feature_names
        
        self.evaluation_results[model_name] = results
        
        # Print summary
        logger.info(f"{model_name} Results:")
        for metric, value in metrics.items():
            if value is not None:
                logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def compare_models(self, models_dict, X_test, y_test, feature_names=None):
        """
        Compare multiple models on the same test set.
        
        Args:
            models_dict (dict): Dictionary of model_name -> model
            X_test (array): Test features
            y_test (array): Test labels
            feature_names (list): Feature names
            
        Returns:
            pd.DataFrame: Comparison table
        """
        comparison_data = []
        
        for model_name, model in models_dict.items():
            results = self.evaluate_single_model(
                model, X_test, y_test, model_name, feature_names
            )
            
            # Extract metrics for comparison
            metrics = results['metrics'].copy()
            metrics['model_name'] = model_name
            comparison_data.append(metrics)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.set_index('model_name')
        
        # Sort by AUC score if available, otherwise by accuracy
        sort_column = 'auc_roc' if 'auc_roc' in comparison_df.columns else 'accuracy'
        comparison_df = comparison_df.sort_values(sort_column, ascending=False)
        
        logger.info("\\nModel Comparison:")
        logger.info(comparison_df.round(4))
        
        return comparison_df
    
    def plot_confusion_matrices(self, models_list=None, class_names=None, figsize=(15, 5)):
        """
        Plot confusion matrices for multiple models.
        
        Args:
            models_list (list): List of model names to plot
            class_names (list): Class names for labels
            figsize (tuple): Figure size
        """
        if models_list is None:
            models_list = list(self.evaluation_results.keys())
        
        n_models = len(models_list)
        fig, axes = plt.subplots(1, n_models, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for i, model_name in enumerate(models_list):
            if model_name in self.evaluation_results:
                cm = self.evaluation_results[model_name]['confusion_matrix']
                
                sns.heatmap(
                    cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[i]
                )
                axes[i].set_title(f'{model_name}\\nConfusion Matrix')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.show()
    
    def plot_roc_curves(self, models_list=None, figsize=(10, 8)):
        """
        Plot ROC curves for multiple models.
        
        Args:
            models_list (list): List of model names to plot
            figsize (tuple): Figure size
        """
        if models_list is None:
            models_list = list(self.evaluation_results.keys())
        
        plt.figure(figsize=figsize)
        
        for model_name in models_list:
            if model_name in self.evaluation_results:
                results = self.evaluation_results[model_name]
                
                if results['probabilities'] is not None:
                    # Get true labels from the first available model
                    y_true = None
                    for _, res in self.evaluation_results.items():
                        if 'y_true' in res:
                            y_true = res['y_true']
                            break
                    
                    if y_true is None:
                        logger.warning("Cannot plot ROC curve: true labels not stored")
                        continue
                    
                    y_pred_proba = results['probabilities']
                    
                    # Handle binary vs multi-class
                    if len(np.unique(y_true)) == 2:
                        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                        auc = roc_auc_score(y_true, y_pred_proba[:, 1])
                        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
                    else:
                        # For multi-class, plot macro-average ROC
                        from sklearn.preprocessing import label_binarize
                        from sklearn.metrics import auc
                        
                        n_classes = len(np.unique(y_true))
                        y_true_bin = label_binarize(y_true, classes=range(n_classes))
                        
                        # Compute macro-average ROC curve
                        fpr = dict()
                        tpr = dict()
                        roc_auc = dict()
                        
                        for i in range(n_classes):
                            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                            roc_auc[i] = auc(fpr[i], tpr[i])
                        
                        # Compute macro-average
                        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
                        mean_tpr = np.zeros_like(all_fpr)
                        
                        for i in range(n_classes):
                            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
                        
                        mean_tpr /= n_classes
                        macro_auc = auc(all_fpr, mean_tpr)
                        
                        plt.plot(all_fpr, mean_tpr, 
                               label=f'{model_name} (Macro AUC = {macro_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def plot_feature_importance(self, model_name, top_n=20, figsize=(10, 8)):
        """
        Plot feature importance for a model.
        
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to plot
            figsize (tuple): Figure size
        """
        if model_name not in self.evaluation_results:
            logger.error(f"Model {model_name} not found in evaluation results")
            return
        
        results = self.evaluation_results[model_name]
        
        if 'feature_importance' not in results:
            logger.warning(f"Feature importance not available for {model_name}")
            return
        
        importance = results['feature_importance']
        feature_names = results.get('feature_names', [f'Feature_{i}' for i in range(len(importance))])
        
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=figsize)
        top_features = importance_df.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance - {model_name}')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return importance_df
    
    def explain_predictions(self, model, X_test, feature_names=None, max_display=10):
        """
        Use SHAP to explain model predictions.
        
        Args:
            model: Trained model
            X_test (array): Test features
            feature_names (list): Feature names
            max_display (int): Maximum features to display
        """
        try:
            # Create SHAP explainer
            if hasattr(model, 'predict_proba'):
                explainer = shap.Explainer(model, X_test[:100])  # Use subset for efficiency
            else:
                explainer = shap.LinearExplainer(model, X_test[:100])
            
            # Calculate SHAP values
            shap_values = explainer(X_test[:100])
            
            # Summary plot
            shap.summary_plot(
                shap_values, 
                X_test[:100], 
                feature_names=feature_names,
                max_display=max_display,
                show=True
            )
            
        except Exception as e:
            logger.error(f"Error generating SHAP explanations: {str(e)}")
            logger.info("SHAP explanations may not be available for all model types")
    
    def plot_learning_curves(self, model, X_train, y_train, cv=5, figsize=(10, 6)):
        """
        Plot learning curves to analyze model performance vs training size.
        
        Args:
            model: Model to evaluate
            X_train (array): Training features
            y_train (array): Training labels
            cv (int): Cross-validation folds
            figsize (tuple): Figure size
        """
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='roc_auc' if len(np.unique(y_train)) == 2 else 'accuracy'
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        plt.figure(figsize=figsize)
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
        
        plt.plot(train_sizes, val_mean, 'o-', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def generate_evaluation_report(self, output_path=None):
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_path (str): Path to save the report
            
        Returns:
            str: Report text
        """
        report = []
        report.append("=" * 60)
        report.append("EXOPLANET CLASSIFICATION - MODEL EVALUATION REPORT")
        report.append("=" * 60)
        report.append()
        
        # Model comparison table
        if len(self.evaluation_results) > 1:
            report.append("MODEL COMPARISON SUMMARY:")
            report.append("-" * 30)
            
            # Create comparison table
            comparison_data = []
            for model_name, results in self.evaluation_results.items():
                metrics = results['metrics'].copy()
                metrics['model_name'] = model_name
                comparison_data.append(metrics)
            
            comparison_df = pd.DataFrame(comparison_data)
            comparison_df = comparison_df.set_index('model_name')
            
            report.append(str(comparison_df.round(4)))
            report.append()
        
        # Detailed results for each model
        for model_name, results in self.evaluation_results.items():
            report.append(f"DETAILED RESULTS - {model_name.upper()}")
            report.append("-" * 40)
            
            # Metrics
            report.append("Metrics:")
            for metric, value in results['metrics'].items():
                if value is not None:
                    report.append(f"  {metric}: {value:.4f}")
            
            report.append()
            
            # Classification report
            report.append("Classification Report:")
            cr = results['classification_report']
            for class_name, metrics in cr.items():
                if isinstance(metrics, dict) and class_name not in ['macro avg', 'weighted avg']:
                    report.append(f"  Class {class_name}:")
                    report.append(f"    Precision: {metrics['precision']:.4f}")
                    report.append(f"    Recall: {metrics['recall']:.4f}")
                    report.append(f"    F1-score: {metrics['f1-score']:.4f}")
            
            report.append()
            report.append("=" * 60)
            report.append()
        
        report_text = "\\n".join(report)
        
        # Save to file if path provided
        if output_path:
            try:
                with open(output_path, 'w') as f:
                    f.write(report_text)
                logger.info(f"Evaluation report saved to {output_path}")
            except Exception as e:
                logger.error(f"Error saving report: {str(e)}")
        
        return report_text


if __name__ == "__main__":
    logger.info("Model evaluation module loaded successfully!")
    
    print("\\nModel Evaluation Features:")
    print("- Comprehensive metrics calculation")
    print("- Multi-model comparison")
    print("- Confusion matrix visualization")
    print("- ROC curve analysis")
    print("- Feature importance plots")
    print("- SHAP explanations")
    print("- Learning curves")
    print("- Detailed evaluation reports")
    
    print("\\nTo use this module:")
    print("1. Train your models using train_tabular.py")
    print("2. Load the evaluator: evaluator = ModelEvaluator()")
    print("3. Evaluate models: evaluator.evaluate_single_model(model, X_test, y_test)")
    print("4. Generate reports and visualizations")