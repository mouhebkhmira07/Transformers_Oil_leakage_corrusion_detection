import os
import json
import logging
import base64
import io
import time
from kafka import KafkaConsumer, KafkaProducer
from ultralytics import YOLO
from PIL import Image
from dotenv import load_dotenv
import cv2
import numpy as np
from pathlib import Path
import sys

# Add src to path to import solutions
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.app.oil_solutions import get_oil_solution, get_severity_color

# Load environment variables
load_dotenv()

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration from environment variables
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(',')
REQUEST_TOPIC = os.getenv('KAFKA_REQUEST_TOPIC', 'oil-detection-requests')
RESULT_TOPIC = os.getenv('KAFKA_RESULT_TOPIC', 'oil-detection-results')
MODEL_PATH = os.getenv('DETECTION_MODEL_PATH', 'models/oil_leakage_yolo.pt')
DEMO_MODE = not os.path.exists(MODEL_PATH)

# Global variables
consumer = None
producer = None
model = None

def load_model():
    """Load YOLOv8 model"""
    global model, DEMO_MODE
    logger.info("🤖 Loading leak/corrosion detection model...")
    
    if DEMO_MODE:
        logger.warning(f"⚠️  Model not found at {MODEL_PATH}")
        logger.warning("⚠️  Running in DEMO MODE - will return simulated results")
        model = None
    else:
        try:
            # Standard PyTorch model
            model = YOLO(MODEL_PATH)
            logger.info("✅ PyTorch Model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading model: {e}")
            logger.warning("⚠️  Falling back to DEMO MODE")
            DEMO_MODE = True
            model = None

def connect_kafka():
    """Connect to Kafka consumer and producer"""
    global consumer, producer
    logger.info("🔌 Connecting to Kafka...")
    
    # Consumer for requests
    # STABILIZER FIX: Added group_id and api_version to prevent flapping
    consumer = KafkaConsumer(
        REQUEST_TOPIC,
        bootstrap_servers=['kafka:29092'], # Internal Docker network address
        group_id='oil-monitor-group',      # CRITICAL: Prevents connection flapping
        auto_offset_reset='earliest',      # Pick up messages even if worker was down
        enable_auto_commit=True,
        value_deserializer=lambda x: json.loads(x.decode('utf-8')),
        api_version=(2, 5, 0)              # Match broker version
    )
    
    # Producer for results
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    logger.info(f"✅ Connected to Kafka")
    logger.info(f"👂 Listening to topic: {REQUEST_TOPIC}")
    logger.info(f"📤 Publishing results to: {RESULT_TOPIC}")

def process_message(message):
    """Process a single inference request"""
    try:
        data = message.value
        request_id = data.get('request_id')
        image_b64 = data.get('image')  # Optional base64 image
        image_path = data.get('image_path')  # Optional path on disk (used by FastAPI producer)
        filename = data.get('filename', 'unknown.jpg')

        logger.info(f"🔄 Processing Request: {request_id} (File: {filename})")

        detected_issues = []
        start_time = time.time()

        # 1. Load / Decode Image
        try:
            if image_b64:
                # Payload contains base64-encoded image
                image_bytes = base64.b64decode(image_b64)
                image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                logger.info("   🖼️  Loaded image from base64 payload")
            elif image_path and os.path.exists(image_path):
                # Payload contains path to image saved by API
                image = Image.open(image_path).convert("RGB")
                logger.info(f"   🖼️  Loaded image from path: {image_path}")
            else:
                raise ValueError("No valid image data found in message (missing 'image' or 'image_path')")
        except Exception as e:
            logger.error(f"❌ Failed to load image: {e}")
            processing_time = time.time() - start_time
            fail_result = {
                "request_id": request_id,
                "success": False,
                "error": f"Failed to load image: {e}",
                "detected_issues": [],
                "timestamp": time.time(),
                "processing_time": processing_time,
            }
            producer.send(RESULT_TOPIC, fail_result)
            producer.flush()
            return

        # 2. Run Inference
        if DEMO_MODE or model is None:
            # DEMO LOGIC
            logger.info("   🎭 Running in demo mode")
            time.sleep(0.5)
            import random
            if random.random() > 0.5:
                # Issue
                key = "oil_leakage" if random.random() > 0.5 else "corrosion"
                sol = get_oil_solution(key)
                if sol:
                    sol['confidence'] = f"{random.uniform(0.7, 0.99):.2f}"
                    sol['severity_color'] = get_severity_color(sol.get('severity'))
                    detected_issues.append(sol)
            else:
                # Healthy
                sol = get_oil_solution("healthy")
                if sol:
                    sol['confidence'] = "1.00"
                    sol['severity_color'] = get_severity_color(sol.get('severity'))
                    detected_issues.append(sol)
        else:
            # REAL INFERENCE
            results = model(image)
            boxes = results[0].boxes
            
            # VISUALIZER: Save annotated image to shared volume
            # Saved to data/shared_results so API can serve it
            annotated_path = f"data/shared_results/annotated_{request_id}.jpg"
            # Ensure directory exists (redundant but safe)
            os.makedirs("data/shared_results", exist_ok=True)
            results[0].save(filename=annotated_path)

            # ACTIVE LEARNING: Review Queue
            REVIEW_DIR = "data/active_learning_queue"
            os.makedirs(REVIEW_DIR, exist_ok=True)

            uncertainty_detected = False

            if len(boxes) == 0:
                # Healthy
                logger.info("   ✅ No objects detected -> Intact Structure")
                sol = get_oil_solution("healthy")
                if sol:
                    sol['confidence'] = "1.00"
                    sol['severity_color'] = get_severity_color(sol.get('severity'))
                    detected_issues.append(sol)
            else:
                # Issues found
                found_classes = set()
                issues_found = []

                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])

                    # ACTIVE LEARNING LOGIC
                    # If model is uncertain (0.25 < conf < 0.55), send to review queue
                    if 0.25 < conf < 0.55:
                        if not uncertainty_detected:  # Save only once per image
                            img_name = f"uncertain_{request_id}.jpg"
                            save_path = os.path.join(REVIEW_DIR, img_name)
                            image.save(save_path)
                            logger.info(f"🚩 ACTIVE LEARNING: Uncertain detection ({conf:.2f}). Saved to {save_path}")
                            uncertainty_detected = True

                    if conf < 0.25:
                        continue

                    c_name = "unknown"
                    if cls_id == 0:
                        c_name = "corrosion"
                    elif cls_id == 1:
                        c_name = "oil_leakage"

                    if c_name != "unknown" and c_name not in found_classes:
                        sol = get_oil_solution(c_name)
                        if sol:
                            sol = sol.copy()
                            sol['confidence'] = f"{conf:.2f}"
                            sol['severity_color'] = get_severity_color(sol.get('severity'))
                            issues_found.append(sol)
                            found_classes.add(c_name)

                if not issues_found:
                    # Low confidence detections ignored -> Healthy
                    sol = get_oil_solution("healthy")
                    if sol:
                        sol['confidence'] = "1.00"
                        sol['severity_color'] = get_severity_color(sol.get('severity'))
                        detected_issues.append(sol)
                else:
                    # Return all unique issues found
                    detected_issues.extend(issues_found)

        processing_time = time.time() - start_time

        # 3. Construct Result
        result = {
            "request_id": request_id,
            "success": True,
            "detected_issues": detected_issues,
            "timestamp": str(time.time()),
            "processing_time": processing_time,
        }

        # 4. Send Result
        producer.send(RESULT_TOPIC, result)
        producer.flush()

        # Log Summary
        issue_names = [i['issue_name'] for i in detected_issues]
        logger.info(f"   📤 Result sent: {issue_names} in {processing_time:.2f}s")

    except Exception as e:
        logger.error(f"❌ Error processing message: {e}")
        processing_time = time.time() - start_time if 'start_time' in locals() else None
        fail_result = {
            "request_id": data.get('request_id') if 'data' in locals() else None,
            "success": False,
            "error": f"Error processing message: {e}",
            "detected_issues": [],
            "timestamp": str(time.time()),
            "processing_time": processing_time,
        }
        try:
            producer.send(RESULT_TOPIC, fail_result)
            producer.flush()
        except Exception:
            # If even publishing the failure result fails, just log it.
            logger.error("❌ Additionally failed to publish error result to Kafka")

def main():
    logger.info("=" * 60)
    logger.info("🚀 Oil & Corrosion Detection Worker Starting...")
    logger.info("=" * 60)
    
    load_model()
    try:
        connect_kafka()
        
        logger.info("✅ Worker is ready.")
        for message in consumer:
            process_message(message)
            
    except KeyboardInterrupt:
        logger.info("👋 Stopping worker...")
    except Exception as e:
        logger.error(f"❌ Critical worker error: {e}")
    finally:
        if consumer: consumer.close()
        if producer: producer.close()

if __name__ == "__main__":
    main()
