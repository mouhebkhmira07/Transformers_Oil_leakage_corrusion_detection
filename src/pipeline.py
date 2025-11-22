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
    mlflow.set_experiment("smart-farming-experiment")

    with mlflow.start_run():
        logging.info("Starting pipeline execution...")
        
        # Log parameters
        mlflow.log_params(params.get('base', {}))
        mlflow.log_params(params.get('data', {}))
        mlflow.log_params(params.get('model', {}))

        # 1. Load Data
        raw_path = params['data']['raw_data_path']
        logging.info(f"Loading raw data from {raw_path}")
        df = pd.read_csv(raw_path)
        
        # 2. Clean Data
        logging.info("Cleaning data...")
        df_cleaned = clean_data(df)
        
        cleaned_path = params['data']['cleaned_data_path']
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
        Path(features_path).parent.mkdir(parents=True, exist_ok=True)
        df_features.to_csv(features_path, index=False)
        logging.info(f"Features saved to {features_path}")
        mlflow.log_artifact(features_path)

        # 4. Train Model (Placeholder)
        logging.info("Model training step (placeholder)...")
        
        # 5. Evaluate Model (Placeholder)
        logging.info("Model evaluation step (placeholder)...")
        
        logging.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLOps pipeline.")
    parser.add_argument("--config", type=str, default="config", help="Path to the config directory")
    args = parser.parse_args()

    run_pipeline(args.config)
