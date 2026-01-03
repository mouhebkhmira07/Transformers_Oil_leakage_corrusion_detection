from ultralytics import YOLO
import mlflow
import os
import yaml
import time
from pathlib import Path
from codecarbon import EmissionsTracker


class _DummyTracker:
    """Fallback tracker used when CodeCarbon/pynvml fails.

    It implements the minimal interface we use (start/stop) so the rest of the
    training code does not need to care whether emissions are tracked or not.
    """

    def start(self):
        return None

    def stop(self):
        # No emissions information available, but keep the return type compatible
        return 0.0


def _create_tracker():
    """Create a CodeCarbon tracker, falling back gracefully if NVML is broken.

    On some Windows setups, `pynvml` and the NVIDIA driver NVML library do not
    match, raising NVMLLibraryMismatchError during initialization. Instead of
    crashing training, we catch any initialization error and continue without
    GPU emissions tracking.
    """
    try:
        tracker = EmissionsTracker(
            project_name="oil_leakage_detection_training",
            output_dir=".",
            output_file="emissions.csv",
            gpu_ids=[],  # Do not monitor any GPUs explicitly
        )
        tracker.start()
        print("🌍 CodeCarbon Emissions Tracker Started...")
        return tracker
    except Exception as e:
        # Typical example: pynvml.NVMLLibraryMismatchError on Windows
        print(f"⚠️ CodeCarbon failed to initialize (GPU tracking disabled): {e}")
        return _DummyTracker()


# CONFIGURATION
def get_config():
    with open("config/params.yaml", "r") as f:
        return yaml.safe_load(f)

def train_detector():
    params = get_config()

    # Initialize CodeCarbon Tracker (with safe fallback)
    tracker = _create_tracker()

    try:
        # Paths from params
        # We need the absolute path to data.yaml
        raw_data_path = Path(params["data"]["raw_data_path"])
        data_yaml_path = raw_data_path / "data.yaml"
        
        MODEL_SAVE_PATH = params["model"]["save_path"]
        
        # 1. Setup MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        mlflow.set_experiment(params["model"]["project"]) # e.g., oil_detection_project (reusing field)

        # 2. Load Detection Model (yolov8n.pt for detection, NOT -cls)
        model_name = "yolov8n.pt" 
        model = YOLO(model_name)

        print(f"🚀 Starting YOLOv8 Training (Detection) on {data_yaml_path}...")
        start_time = time.time()

        with mlflow.start_run():
            # 3. Train
            # Note: 'data' argument in YOLO detection mode points to the data.yaml file
            results = model.train(
                data=str(data_yaml_path.absolute()),
                epochs=params["model"]["epochs"],
                imgsz=params["model"]["imgsz"],
                batch=params["model"]["batch"],
                project="oil_project",
                name="oil_detector_experiment"
            )
            
            training_time = time.time() - start_time
            
            # 4. Export to ONNX (Optional but good for MLOps)
            
            # 5. Save the best pytorch model to our models/ folder
            # YOLO saves best.pt in runs/detect/train/weights/best.pt
            # We want to move it to models/oil_leakage_yolo.pt
            
            # Determine output directory
            # Ultralytics returns the save dir in results.save_dir causing issues depending on version
            # We can usually rely on the project/name structure
            # Standard path: oil_project/oil_detector_experiment/weights/best.pt
            
            run_dir = Path("oil_project") / "oil_detector_experiment"
            # If run exists, it might append a number (oil_detector_experiment2), so let's check results.save_dir if available
            if hasattr(results, 'save_dir'):
                 run_dir = Path(results.save_dir)
                 
            best_weight_path = run_dir / "weights" / "best.pt"
            
            if best_weight_path.exists():
                # Create models dir if not exists
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                
                # Simple copy/rename
                # On Windows, rename across drives/mounts can fail, so read/write is safer or shutil
                import shutil
                shutil.copy2(best_weight_path, MODEL_SAVE_PATH)
                
                print(f"✅ Best model saved to {MODEL_SAVE_PATH}")
                mlflow.log_artifact(MODEL_SAVE_PATH)
            else:
                print(f"⚠️ Could not find best.pt at {best_weight_path}. Check 'oil_project' folder.")

            # Log Metrics
            mlflow.log_metric("training_duration_seconds", training_time)
            
            print(f"✅ Training Complete. Duration: {training_time:.2f}s")
            
    except Exception as e:
        print(f"❌ Error during training: {e}")
        raise e
    finally:
        emissions = tracker.stop()
        print(f"🌍 Training Emissions: {emissions:.4f} kg CO2eq")

if __name__ == "__main__":
    train_detector()
