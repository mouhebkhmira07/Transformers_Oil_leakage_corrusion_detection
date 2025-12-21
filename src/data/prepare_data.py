import yaml
import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_data():
    """
    Fixes the paths in data.yaml to be absolute paths for YOLOv8 training.
    """
    # Load params
    with open("config/params.yaml", "r") as f:
        params = yaml.safe_load(f)
        
    raw_data_path = Path(params["data"]["raw_data_path"])
    data_yaml_path = raw_data_path / "data.yaml"
    
    if not data_yaml_path.exists():
        logging.error(f"❌ data.yaml not found at {data_yaml_path}")
        return
        
    # Read current data.yaml
    with open(data_yaml_path, "r") as f:
        data_config = yaml.safe_load(f)
        
    logging.info(f"📄 Current configuration: {data_config}")
    
    # Update paths to absolute
    # Roboflow usually outputs 'train', 'valid', 'test' folders relative to data.yaml
    # We want to be explicit
    
    base_path = raw_data_path.absolute()
    
    data_config['path'] = str(base_path)
    data_config['train'] = "train/images"
    data_config['val'] = "valid/images"
    data_config['test'] = "test/images"
    
    # Write back
    with open(data_yaml_path, "w") as f:
        yaml.dump(data_config, f, default_flow_style=False)
        
    logging.info(f"✅ params.yaml modified and saved to {data_yaml_path}")
    logging.info(f"   path: {data_config['path']}")
    logging.info(f"   train: {data_config['train']}")
    logging.info(f"   val: {data_config['val']}")
    
    # Create output marker for DVC
    marker_path = Path("data/prepared.lock")
    marker_path.touch()
    logging.info(f"✅ Created marker file at {marker_path}")

if __name__ == "__main__":
    prepare_data()
