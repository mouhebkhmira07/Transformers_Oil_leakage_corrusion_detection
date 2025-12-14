import argparse
import os
import yaml
import json
import logging
import pandas as pd
from pathlib import Path
import sys
import mlflow

# Add src to path to ensure imports work if run from root
sys.path.append(os.path.join(os.getcwd(), 'src'))

from data.make_dataset import clean_data
from features.build_features import add_all_agriculture_features

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params(params_path):
    """Load parameters from a YAML file."""
    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Params file not found: {params_path}")
    with open(params_path, 'r') as f:
        params = yaml.safe_load(f)
    logging.info(f"Parameters loaded from {params_path}")
    return params

def load_schema(schema_path):
    """Load schema from a JSON file."""
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    logging.info(f"Schema loaded from {schema_path}")
    return schema

def run_pipeline(config_dir):
    """Run the MLOps pipeline."""
    params_path = os.path.join(config_dir, 'params.yaml')
    schema_path = os.path.join(config_dir, 'schema.json')

    params = load_params(params_path)
    schema = load_schema(schema_path)

    # Set MLflow experiment
    experiment_name = "smart-farming-experiment-wsl"
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        logging.info("Starting pipeline execution...")
        
        # Log parameters
        mlflow.log_params(params.get('base', {}))
        mlflow.log_params(params.get('data', {}))
        mlflow.log_params(params.get('model', {}))

        # 1. Load Data
        raw_path = params['data']['raw_data_path']
        # Resolve raw_path relative to project root if it's relative
        if not os.path.isabs(raw_path):
            raw_path = os.path.join(config_dir, "..", raw_path)
            
        logging.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)
        
        # 2. Clean Data
        logging.info("Cleaning data...")
        df_cleaned = clean_data(df)
        
        cleaned_path = params['data']['cleaned_data_path']
        if not os.path.isabs(cleaned_path):
            cleaned_path = os.path.join(config_dir, "..", cleaned_path)

        Path(cleaned_path).parent.mkdir(parents=True, exist_ok=True)
        df_cleaned.to_csv(cleaned_path, index=False)
        logging.info(f"Cleaned data saved to {cleaned_path}")
        mlflow.log_artifact(cleaned_path)

        # 3. Build Features
        logging.info("Building features...")
        # Ensure datetime columns are parsed before feature engineering
        for col in ['timestamp', 'sowing_date', 'harvest_date']:
            if col in df_cleaned.columns:
                df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                
        df_features = add_all_agriculture_features(df_cleaned)
        
        features_path = params['data']['features_path']
        if not os.path.isabs(features_path):
            features_path = os.path.join(config_dir, "..", features_path)

        Path(features_path).parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(features_path, index=False)
        logging.info(f"Features saved to {features_path}")
        mlflow.log_artifact(features_path)

        # 4. Train Model
        logging.info("Training models...")
        # We need to split the data first if not already done, but build_features does it now.
        # However, build_features saves to train.csv and test.csv.
        # Let's assume they are in the same directory as features_path
        processed_dir = Path(features_path).parent
        train_path = processed_dir / "train.csv"
        test_path = processed_dir / "test.csv"
        models_dir = Path(config_dir).parent / "models"

        # Check if train/test exist, if not, we might need to rely on build_features having run
        if not train_path.exists() or not test_path.exists():
             logging.warning("Train/Test files not found. Relying on build_features to have created them.")
        
        # We can call the click command programmatically or just import the logic. 
        # Calling via CliRunner or subprocess is safer for click apps if we don't want to refactor them.
        # Or better, let's just use subprocess to run them as scripts to avoid context issues.
        import subprocess
        
        # Train
        subprocess.run(["python", "src/models/train_model.py", str(train_path), str(models_dir)], check=True)
        
        # 5. Evaluate Model
        logging.info("Evaluating models...")
        subprocess.run(["python", "src/models/evaluate_model.py", str(test_path), str(models_dir), str(Path(config_dir).parent / "reports")], check=True)
        
        logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    # Determine project root relative to this script
    # src/pipeline.py -> parent is src -> parent is project root
    project_root = Path(__file__).resolve().parent.parent
    
    # Add src to sys.path
    src_path = project_root / 'src'
    if str(src_path) not in sys.path:
        sys.path.append(str(src_path))

    parser = argparse.ArgumentParser(description="Run the MLOps pipeline.")
    parser.add_argument("--config", type=str, default="config", help="Path to the config directory")
    args = parser.parse_args()

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Try relative to current working directory first
        if not config_path.exists():
            # Try relative to project root
            config_path = project_root / args.config
            
    if not config_path.exists():
        raise FileNotFoundError(f"Config directory not found at {args.config} or {config_path}")

    run_pipeline(str(config_path))
