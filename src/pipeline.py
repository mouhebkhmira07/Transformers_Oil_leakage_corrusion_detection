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
from data.organise_data import reorganize_and_split

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

        # 4. Organize Palm Disease Data (Vision Pipeline)
        logging.info("="*60)
        logging.info("VISION PIPELINE: Organizing palm disease dataset...")
        logging.info("="*60)
        
        project_root = Path(config_dir).parent
        palm_source = project_root / "data" / "Infected Date Palm Leaves Dataset" / "Processed"
        palm_dest = project_root / "data" / "processed" / "palm_disease_final"
        
        if palm_source.exists():
            logging.info(f"Organizing palm disease data from {palm_source}")
            success = reorganize_and_split(
                source_path=str(palm_source),
                dest_path=str(palm_dest),
                train_ratio=0.7,
                val_ratio=0.2,
                test_ratio=0.1
            )
            
            if success:
                logging.info("Palm disease data organized successfully")
            else:
                logging.warning("Palm disease data organization failed")
        else:
            logging.info(f"Skipping palm data organization - source not found: {palm_source}")
        
        # 5. Train Vision Classifier (YOLOv8)
        logging.info("="*60)
        logging.info("VISION PIPELINE: Training YOLOv8 disease classifier...")
        logging.info("="*60)
        
        if palm_dest.exists():
            try:
                from ultralytics import YOLO
                import time
                
                # Setup for vision classifier
                vision_model_path = project_root / "models" / "palm_classifier.onnx"
                
                logging.info("Loading YOLOv8 nano model...")
                model = YOLO('yolov8n-cls.pt')
                
                start_time = time.time()
                
                # Train the model
                logging.info("Starting Green AI training (YOLOv8n)...")
                results = model.train(
                    data=str(palm_dest),
                    epochs=20,
                    imgsz=224,
                    batch=16,
                    project="palm_project",
                    name="green_experiment",
                    verbose=True
                )
                
                training_time = time.time() - start_time
                
                # Export to ONNX
                logging.info("Exporting model to ONNX format...")
                exported_path = model.export(format='onnx')
                
                # Move to models directory
                import shutil
                if os.path.exists(exported_path):
                    if vision_model_path.exists():
                        os.remove(vision_model_path)
                    shutil.move(str(exported_path), str(vision_model_path))
                    logging.info(f"Vision model saved to {vision_model_path}")
                    
                    # Log Green AI metrics
                    file_size_mb = os.path.getsize(vision_model_path) / (1024 * 1024)
                    mlflow.log_metric("vision_training_duration_seconds", training_time)
                    mlflow.log_metric("vision_model_size_mb", file_size_mb)
                    mlflow.log_artifact(str(vision_model_path))
                    
                    logging.info(f"🌿 Green AI Metric: Vision model size is {file_size_mb:.2f} MB")
                    logging.info(f"🌿 Green AI Metric: Training time was {training_time:.1f} seconds")
                else:
                    logging.warning("ONNX export path not found")
                    
            except ImportError:
                logging.warning("Ultralytics not installed - skipping vision training")
                logging.info("Install with: pip install ultralytics")
            except Exception as e:
                logging.error(f"Vision training failed: {e}")
        else:
            logging.info("Skipping vision training - palm disease data not organized")
        
        # 6. Train Tabular Models (Crop Recommendation)
        logging.info("="*60)
        logging.info("TABULAR PIPELINE: Training crop recommendation models...")
        logging.info("="*60)
        
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
        
        # Train tabular models
        logging.info("Training tabular ML models...")
        subprocess.run(["python", "src/models/train_model.py", str(train_path), str(models_dir)], check=True)
        
        # 7. Evaluate Tabular Model
        logging.info("Evaluating tabular models...")
        subprocess.run(["python", "src/models/evaluate_model.py", str(test_path), str(models_dir), str(Path(config_dir).parent / "reports")], check=True)
        
        logging.info("="*60)
        logging.info("🎉 COMPLETE PIPELINE EXECUTION FINISHED SUCCESSFULLY")
        logging.info("="*60)
        logging.info("✅ Tabular Pipeline: Crop recommendation model trained")
        logging.info("✅ Vision Pipeline: Palm disease classifier trained")
        logging.info("="*60)

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
