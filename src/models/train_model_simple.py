"""
Train Machine Learning Models for Crop Recommendation

A streamlined training script that:
- Trains multiple ML models (RandomForest, XGBoost, KNN, SVM)
- Selects the best performer based on training accuracy
- Logs all experiments to MLflow
- Saves models locally for deployment
"""

import click
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from mlflow.models.signature import infer_signature


def log_msg(msg):
    """Print formatted log message."""
    print(f"--> {msg}")


def train_and_log(model, X, y, model_name, output_dir):
    """
    Train a model and log it to MLflow.
    
    Args:
        model: Scikit-learn compatible model
        X: Feature matrix
        y: Target vector
        model_name: Name for MLflow run
        output_dir: Directory to save model file
    """
    with mlflow.start_run(run_name=model_name):
        log_msg(f"Logging {model_name}...")
        
        # Train the model
        model.fit(X, y)
        
        # Infer signature for MLflow model registry
        signature = infer_signature(X, model.predict(X))
        
        # Log parameters and model to MLflow
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(model, model_name, signature=signature)
        
        # Save model locally
        local_path = output_dir / f"{model_name}.pkl"
        joblib.dump(model, local_path)
        log_msg(f"Saved {local_path}")


def train_crop_recommender(train_df, output_dir):
    """
    Train crop recommendation models in a tournament-style comparison.
    
    Args:
        train_df: Training dataframe with features and target
        output_dir: Directory to save trained models
    """
    log_msg("--- Training Crop Recommender Tournament ---")
    
    # Define target and features (lowercase to match processed data)
    target = 'label'
    features = ['n', 'p', 'k', 'temperature', 'humidity', 'ph', 'rainfall']
    
    # Validate that columns exist
    if target not in train_df.columns:
        log_msg(f"ERROR: Target '{target}' not found. Columns are: {train_df.columns.tolist()}")
        return
    
    missing_feats = [f for f in features if f not in train_df.columns]
    if missing_feats:
        log_msg(f"ERROR: Missing features: {missing_feats}")
        return

    # Prepare data
    X = train_df[features]
    y = train_df[target]
    
    # Encode target (crop names -> numbers)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Save the encoder (crucial for evaluation and deployment)
    encoder_path = output_dir / "crop_label_encoder.pkl"
    joblib.dump(le, encoder_path)
    log_msg(f"Saved crop_label_encoder.pkl with {len(le.classes_)} classes")

    # Define the tournament contenders
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "SVM": SVC(probability=True, random_state=42)
    }

    # Tournament: train all models and track the best
    best_score = -1
    best_model_obj = None
    best_name = None

    for name, model in models.items():
        try:
            # Train and evaluate
            model.fit(X, y_encoded)
            acc = model.score(X, y_encoded)
            log_msg(f"[{name}] Train Accuracy: {acc:.4f}")
            
            # Save individual model version
            train_and_log(model, X, y_encoded, f"crop_recommender_{name}", output_dir)
            
            # Track the champion
            if acc > best_score:
                best_score = acc
                best_model_obj = model
                best_name = name
                
        except Exception as e:
            log_msg(f"[{name}] Failed: {e}")

    # Save the champion as the default model
    if best_model_obj:
        log_msg(f"🏆 CHAMPION: {best_name} (Accuracy: {best_score:.4f})")
        joblib.dump(best_model_obj, output_dir / "crop_recommender.pkl")
        log_msg("Saved champion as crop_recommender.pkl")
    else:
        log_msg("ERROR: No models trained successfully")


def train_disease_detector(train_df, output_dir):
    """
    Train disease detection model (placeholder for future dataset).
    
    Args:
        train_df: Training dataframe
        output_dir: Directory to save trained models
    """
    target = 'crop_disease_status'
    
    # Check if disease data exists
    if target not in train_df.columns:
        log_msg("No disease target found. Skipping Disease Detector.")
        return

    # Placeholder for future implementation
    log_msg("Disease Detector logic placeholder - implement when dataset is available.")


@click.command()
@click.argument('train_data_path', type=click.Path(exists=True))
@click.argument('models_dir', type=click.Path())
def main(train_data_path, models_dir):
    """
    Train crop recommendation and disease detection models.
    
    Usage:
        python train_model.py data/processed/train.csv models/
    """
    log_msg("=" * 60)
    log_msg("STARTING TRAINING SCRIPT")
    log_msg("=" * 60)
    
    # Setup output directory
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # Load training data
    log_msg(f"Loading data from {train_data_path}...")
    train_df = pd.read_csv(train_data_path)
    log_msg(f"Loaded {len(train_df)} rows, {len(train_df.columns)} columns")
    
    # Train models
    train_crop_recommender(train_df, models_path)
    train_disease_detector(train_df, models_path)
    
    log_msg("=" * 60)
    log_msg("TRAINING FINISHED")
    log_msg("=" * 60)


if __name__ == '__main__':
    main()
