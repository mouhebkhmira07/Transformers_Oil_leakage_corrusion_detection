from fastapi import FastAPI, HTTPException, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import json
import uuid
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import logging
from datetime import datetime
import io
from PIL import Image
import aiofiles
from pathlib import Path

# Import oil detection solutions
from .oil_solutions import get_oil_solution, get_all_issues

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global Kafka producer and consumer
kafka_producer: Optional[AIOKafkaProducer] = None
kafka_consumer: Optional[AIOKafkaConsumer] = None

# In-memory result store (in production, use Redis or database)
result_store: Dict[str, dict] = {}

# Image upload directory
UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    global kafka_producer, kafka_consumer
    
    # Startup
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    request_topic = os.getenv("KAFKA_REQUEST_TOPIC", "oil-detection-requests")
    result_topic = os.getenv("KAFKA_RESULT_TOPIC", "oil-detection-results")
    
    try:
        # Initialize Kafka producer for requests
        kafka_producer = AIOKafkaProducer(
            bootstrap_servers=kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await kafka_producer.start()
        logger.info(f"✅ Kafka producer started - publishing to {request_topic}")
        
        # Initialize Kafka consumer for results
        kafka_consumer = AIOKafkaConsumer(
            result_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset='latest',
            enable_auto_commit=True,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        await kafka_consumer.start()
        logger.info(f"✅ Kafka consumer started - listening to {result_topic}")
        
        # Start background task to consume results
        import asyncio
        asyncio.create_task(consume_results())
        
    except Exception as e:
        logger.error(f"❌ Failed to start Kafka: {e}")
        kafka_producer = None
        kafka_consumer = None
    
    yield
    
    # Shutdown
    if kafka_producer:
        await kafka_producer.stop()
        logger.info("Kafka producer stopped")
    if kafka_consumer:
        await kafka_consumer.stop()
        logger.info("Kafka consumer stopped")

app = FastAPI(
    lifespan=lifespan, 
    title="Oil Leakage and Corrosion Detection API", 
    version="3.0.0",
    description="Upload infrastructure images - API publishes to Kafka, worker processes for leakage and corrosion"
)

# Static files and Visualizer
app.mount("/static", StaticFiles(directory="src/app/static"), name="static")
app.mount("/outputs", StaticFiles(directory="data/shared_results"), name="outputs")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    return FileResponse("src/app/static/index.html")

class DetectionRequest(BaseModel):
    """Response model for detection request"""
    success: bool
    request_id: str
    message: str
    timestamp: str
    status_url: str

class DetectionResult(BaseModel):
    """Response model for detection result"""
    request_id: str
    status: str  # pending, processing, completed, failed
    timestamp: str
    detected_issues: Optional[List[dict]] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None

async def consume_results():
    """Background task to consume results from Kafka"""
    logger.info("🎧 Result consumer background task started")
    
    if not kafka_consumer:
        logger.warning("⚠️  Kafka consumer not initialized, result collection disabled")
        return
    
    try:
        async for message in kafka_consumer:
            result = message.value
            request_id = result.get('request_id')
            
            if request_id:
                result['status'] = 'completed' if result.get('success') else 'failed'
                result_store[request_id] = result
                logger.info(f"📥 Received result for request: {request_id}")
    except Exception as e:
        logger.error(f"❌ Error in result consumer: {e}")

async def save_uploaded_image(file: UploadFile) -> str:
    """Save uploaded image to disk and return path"""
    # Generate unique filename
    file_ext = Path(file.filename).suffix or '.jpg'
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    file_path = UPLOAD_DIR / unique_filename
    
    # Save file
    async with aiofiles.open(file_path, 'wb') as f:
        content = await file.read()
        await f.write(content)
    
    return str(file_path)

@app.post("/detect", response_model=DetectionRequest)
async def detect_leakage(file: UploadFile = File(...)):
    """
    Upload an infrastructure image for oil leakage or corrosion detection (async)
    
    This endpoint:
    1. Saves the uploaded image
    2. Publishes a request to Kafka
    3. Returns immediately with request_id
    4. Worker processes the request asynchronously
    
    Use /result/{request_id} to check status and get results
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Save image
        image_path = await save_uploaded_image(file)
        logger.info(f"📁 Saved image: {image_path}")
        
        # Prepare request message
        request_message = {
            'request_id': request_id,
            'image_path': image_path,
            'filename': file.filename,
            'timestamp': timestamp
        }
        
        # Initialize result in store
        result_store[request_id] = {
            'request_id': request_id,
            'status': 'pending',
            'timestamp': timestamp,
            'filename': file.filename
        }
        
        # Publish to Kafka
        if kafka_producer:
            request_topic = os.getenv("KAFKA_REQUEST_TOPIC", "oil-detection-requests")
            try:
                await kafka_producer.send_and_wait(request_topic, request_message)
                logger.info(f"📤 Published request {request_id} to {request_topic}")
                
                return DetectionRequest(
                    success=True,
                    request_id=request_id,
                    message="Request received and queued for processing",
                    timestamp=timestamp,
                    status_url=f"/result/{request_id}"
                )
            except Exception as e:
                logger.error(f"❌ Failed to publish to Kafka: {e}")
                raise HTTPException(status_code=503, detail="Message queue unavailable")
        else:
            raise HTTPException(status_code=503, detail="Kafka producer not available")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Detection request error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/result/{request_id}", response_model=DetectionResult)
async def get_result(request_id: str):
    """
    Get the result of a detection request
    
    Status values:
    - pending: Request received, waiting for worker
    - processing: Worker is processing the image
    - completed: Processing done, results available
    - failed: Processing failed
    """
    if request_id not in result_store:
        raise HTTPException(status_code=404, detail="Request ID not found")
    
    result = result_store[request_id]
    
    return DetectionResult(
        request_id=result['request_id'],
        status=result['status'],
        timestamp=result['timestamp'],
        detected_issues=result.get('detected_issues'),
        processing_time=result.get('processing_time'),
        error=result.get('error')
    )

@app.get("/issues")
async def list_issues():
    """Get list of all detectable issues with their information"""
    issues = get_all_issues()
    issue_info = []
    
    for issue_key in issues:
        solution = get_oil_solution(issue_key)
        issue_info.append({
            "key": issue_key,
            "name": solution.get("issue_name", issue_key),
            "severity": solution.get("severity", "unknown"),
            "description": solution.get("description", "")
        })
    
    return {
        "total_issues": len(issues),
        "issues": issue_info
    }

# Note: Root route moved to serve_frontend to serve the UI

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "kafka_producer_connected": kafka_producer is not None,
        "kafka_consumer_connected": kafka_consumer is not None,
        "upload_dir": str(UPLOAD_DIR),
        "pending_results": len([r for r in result_store.values() if r['status'] == 'pending']),
        "timestamp": datetime.utcnow().isoformat()
    }
