import os
import yaml
import requests
import zipfile
import io
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_data_direct():
    # Load params
    try:
        with open("config/params.yaml", "r") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        # Fallback if running from src/data
        with open("../../config/params.yaml", "r") as f:
            params = yaml.safe_load(f)

    # Configuration
    # URL provided by user
    DOWNLOAD_URL = "https://app.roboflow.com/ds/Tpza3of8tp?key=VuumCoK1dd"
    
    raw_data_path = Path(params["data"]["raw_data_path"])
    
    # Clean and create directory
    if raw_data_path.exists():
        shutil.rmtree(raw_data_path)
    raw_data_path.mkdir(parents=True, exist_ok=True)
    
    logging.info(f"📂 Target Directory: {raw_data_path.absolute()}")
    logging.info("🚀 Starting Direct Download (cURL equivalent)...")
    
    try:
        zip_path = raw_data_path / "roboflow_data.zip"
        
        if not zip_path.exists():
            # Download
            logging.info("⬇️  Downloading zip file...")
            response = requests.get(DOWNLOAD_URL, stream=True)
            response.raise_for_status()
            
            with open(zip_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    f.write(chunk)
            logging.info("✅ Download complete.")

        # Extract with renaming for long files
        logging.info("📦 Extracting zip file with safe renaming...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            for file_info in z.infolist():
                # Get the original path parts
                parts = file_info.filename.split('/')
                
                # Check basename (last part)
                basename = parts[-1]
                if len(basename) > 50:
                    # Rename to short hash
                    import hashlib
                    name, ext = os.path.splitext(basename)
                    hash_name = hashlib.md5(name.encode()).hexdigest()[:10]
                    new_basename = f"{hash_name}{ext}"
                    parts[-1] = new_basename
                    
                    logging.info(f"  Renaming {basename[:20]}... to {new_basename}")
                
                # Reconstruct path
                new_filename = os.path.join(*parts)
                target_path = raw_data_path / new_filename
                
                # Create directories
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Extract file
                if not file_info.is_dir():
                    with open(target_path, "wb") as f:
                        f.write(z.read(file_info))
            
        logging.info(f"✅ Extracted to {raw_data_path}")
       
        logging.info(f"📂 Contents of {raw_data_path}:")
        for item in raw_data_path.iterdir():
            logging.info(f"  - {item.name}")
            
    except Exception as e:
        logging.error(f"❌ Failed to download/extract: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    download_data_direct()
