"""
Evaluate Machine Learning Models for Crop Recommendation

This script evaluates trained models on test data and generates:
- Classification metrics (accuracy, precision, recall, F1)
- Confusion matrices with visualizations
- MLflow experiment tracking
- JSON reports for downstream analysis
"""

import click
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def log_msg(msg, level="INFO"):
    """Print formatted log message."""
    prefix = {
        "INFO": "ℹ️",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "SUCCESS": "✅"
    }.get(level, "→")
    print(f"{prefix} {msg}")


def plot_confusion_matrix(y_true, y_pred, labels, output_path, title="Confusion Matrix"):
    """Create and save a confusion matrix visualization."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_msg(f"Saved confusion matrix to {output_path}", "SUCCESS")


def evaluate_crop_recommender(test_df, model_path, output_dir, models_dir):
    """
    Evaluate crop recommendation model on test data.
    
    This function correctly loads the label encoder and decodes predictions
    to prevent the "0 vs Rice" mismatch.
    """
    log_msg("=" * 60, "INFO")
    log_msg("EVALUATING CROP RECOMMENDER", "INFO")
    log_msg("=" * 60, "INFO")
    
    # Load model
    model = joblib.load(model_path)
    log_msg(f"Loaded model from {model_path}", "SUCCESS")
    
    # Load label encoder (CRITICAL for decoding predictions)
    encoder_path = Path(models_dir) / "crop_label_encoder.pkl"
    
    if not encoder_path.exists():
        log_msg("CRITICAL ERROR: crop_label_encoder.pkl not found!", "ERROR")
        log_msg("Cannot decode predictions without the encoder.", "ERROR")
        return
    
    le = joblib.load(encoder_path)
    log_msg(f"Loaded label encoder with {len(le.classes_)} classes", "SUCCESS")

    # Define features and target (UPPERCASE N, P, K to match processed data)
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'
    
    # Validate columns exist
    missing_features = [f for f in features if f not in test_df.columns]
    if missing_features:
        log_msg(f"Missing features: {missing_features}", "ERROR")
        return
    
    if target not in test_df.columns:
        log_msg(f"Target column '{target}' not found", "ERROR")
        return
    
    # Prepare test data
    X_test = test_df[features]
    y_true = test_df[target]  # String labels (e.g., "Rice", "Wheat")
    
    log_msg(f"Test set size: {len(X_test)} samples", "INFO")
    
    # Make predictions
    y_pred_numeric = model.predict(X_test)  # Numeric labels (e.g., 0, 1, 2)
    
    # CRITICAL: Decode numeric predictions to crop names
    y_pred = le.inverse_transform(y_pred_numeric)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    log_msg(f"Test Accuracy:  {accuracy:.4f}", "SUCCESS")
    log_msg(f"Precision:      {precision:.4f}", "INFO")
    log_msg(f"Recall:         {recall:.4f}", "INFO")
    log_msg(f"F1 Score:       {f1:.4f}", "INFO")
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Save metrics as JSON
    metrics_path = output_dir / "metrics_crop_recommender.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    log_msg(f"Saved metrics to {metrics_path}", "SUCCESS")
    
    # Generate confusion matrix
    cm_path = output_dir / "confusion_matrix_crop_recommender.png"
    plot_confusion_matrix(
        y_true, 
        y_pred, 
        labels=sorted(le.classes_),
        output_path=cm_path,
        title="Crop Recommender Confusion Matrix"
    )
    
    # Log to MLflow
    try:
        with mlflow.start_run(run_name="crop_recommender_evaluation"):
            mlflow.log_metrics({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1_score": f1
            })
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(cm_path))
            log_msg("Logged metrics to MLflow", "SUCCESS")
    except Exception as e:
        log_msg(f"MLflow logging failed: {e}", "WARNING")


def evaluate_disease_detector(test_df, model_path, output_dir):
    """Evaluate disease detection model on test data."""
    log_msg("=" * 60, "INFO")
    log_msg("EVALUATING DISEASE DETECTOR", "INFO")
    log_msg("=" * 60, "INFO")
    
    model = joblib.load(model_path)
    log_msg(f"Loaded model from {model_path}", "SUCCESS")
    
    features = ['temperature_c', 'humidity_%', 'soil_moisture_%', 'ndvi_index']
    target = 'crop_disease_status'
    
    # Check if required columns exist
    missing_features = [f for f in features if f not in test_df.columns]
    if missing_features or target not in test_df.columns:
        log_msg(
            f"Missing columns: {missing_features + ([target] if target not in test_df.columns else [])}",
            "WARNING"
        )
        log_msg("Skipping disease detector evaluation", "WARNING")
        return
    
    # Clean data
    test_df_clean = test_df.dropna(subset=[target])
    
    if len(test_df_clean) == 0:
        log_msg("No valid test samples after removing NaN", "WARNING")
        return
    
    X_test = test_df_clean[features]
    y_true = test_df_clean[target]
    
    log_msg(f"Test set size: {len(X_test)} samples", "INFO")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    log_msg(f"Test Accuracy: {accuracy:.4f}", "SUCCESS")
    
    # Classification report
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Save metrics
    metrics_path = output_dir / "metrics_disease_detector.json"
    with open(metrics_path, "w") as f:
        json.dump(report, f, indent=4)
    log_msg(f"Saved metrics to {metrics_path}", "SUCCESS")
    
    # Generate confusion matrix
    try:
        cm_path = output_dir / "confusion_matrix_disease_detector.png"
        labels = sorted(test_df_clean[target].unique())
        plot_confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            output_path=cm_path,
            title="Disease Detector Confusion Matrix"
        )
    except Exception as e:
        log_msg(f"Could not generate confusion matrix: {e}", "WARNING")
    
    # Log to MLflow
    try:
        with mlflow.start_run(run_name="disease_detector_evaluation"):
            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_artifact(str(metrics_path))
            if cm_path.exists():
                mlflow.log_artifact(str(cm_path))
            log_msg("Logged metrics to MLflow", "SUCCESS")
    except Exception as e:
        log_msg(f"MLflow logging failed: {e}", "WARNING")


@click.command()
@click.argument('test_data_path', type=click.Path(exists=True))
@click.argument('models_dir', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path())
@click.option('--experiment-name', default='crop_farming_models',
              help='MLflow experiment name')
def main(test_data_path, models_dir, output_dir, experiment_name='crop_farming_models'):
    """
    Evaluate trained models on test data.
    
    Usage:
        python evaluate_model.py data/processed/test.csv models/ reports/
    """
    log_msg("=" * 60, "INFO")
    log_msg("MODEL EVALUATION PIPELINE", "INFO")
    log_msg("=" * 60, "INFO")
    
    # Setup
    models_path = Path(models_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load test data
    log_msg(f"Loading test data from {test_data_path}...", "INFO")
    test_df = pd.read_csv(test_data_path)
    log_msg(f"Loaded {len(test_df)} test samples", "SUCCESS")
    
    # Evaluate Crop Recommender
    recommender_path = models_path / "crop_recommender.pkl"
    if recommender_path.exists():
        evaluate_crop_recommender(test_df, recommender_path, out_path, models_path)
    else:
        log_msg("crop_recommender.pkl not found, skipping", "WARNING")
    
    # Evaluate Disease Detector
    disease_path = models_path / "disease_detector.pkl"
    if disease_path.exists():
        evaluate_disease_detector(test_df, disease_path, out_path)
    else:
        log_msg("disease_detector.pkl not found, skipping", "WARNING")
    
    log_msg("=" * 60, "INFO")
    log_msg("EVALUATION COMPLETED", "SUCCESS")
    log_msg("=" * 60, "INFO")


if __name__ == '__main__':
    main()