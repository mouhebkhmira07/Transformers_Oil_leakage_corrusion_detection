# Oil Leakage and Corrosion Detection - DLOps

![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLOv8-blue)
![MLflow](https://img.shields.io/badge/MLflow-Tracking-orange)
![DVC](https://img.shields.io/badge/DVC-Pipeline-green)
![Kafka](https://img.shields.io/badge/Kafka-Streaming-black)

A production-ready Deep Learning Operations (DLOps) project for detecting and classifying oil leakage and surface corrosion in industrial infrastructure using YOLOv8 computer vision models, with MLflow experiment tracking, DVC pipeline management, and an asynchronous Producer-Consumer architecture via Kafka.

## 🎯 Project Overview

This project implements an end-to-end computer vision pipeline for monitoring industrial infrastructure for leaks and corrosion. It emphasizes:

- **Computer Vision**: YOLOv8-based detection and classification of leakage and corrosion.
- **Asynchronous Architecture**: Producer-Consumer pattern using Kafka for high-throughput image processing.
- **MLOps Best Practices**: DVC pipelines, MLflow tracking, Dockerized environment.
- **Scalable Infrastructure**: Microservices for API and Inference tasks.

## 📁 Project Structure

```
├── data/
│   ├── raw/                             # Raw image dataset
│   ├── processed/                       # Organized train/val/test splits
│   └── uploads/                         # Temporary storage for API uploads
├── models/
│   └── oil_leakage_yolo.pt              # Trained YOLOv8 model
├── src/
│   ├── app/                             # FastAPI Producer application
│   │   ├── main.py                      # API endpoints and Kafka producer
│   │   └── oil_solutions.py             # Issue database and mitigation plans
│   ├── consumer/                        # Kafka Inference Worker
│   │   └── inference_worker.py          # YOLO inference and result publishing
│   ├── data/                            # Data processing scripts
│   └── vision/                          # Training scripts
├── docker-compose.yaml                  # Infrastructure orchestration
├── dvc.yaml                             # DVC pipeline definition
└── .github/workflows/                   # CI/CD configuration
```

## 🚀 Getting Started

### Prerequisites

- Docker & Docker Compose
- Python 3.9+
- Git

### Running the Application

1. **Start the infrastructure**
```bash
docker-compose up -d
```
This starts:
- **Zookeeper & Kafka**: For message streaming.
- **FastAPI App**: The Producer API (Port 8000).
- **Inference Worker**: The Consumer processing images.

2. **Access the API**
- Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

### Using the API

1. **Upload an image**
```bash
POST /detect
Content-Type: multipart/form-data
File: <image_file>
```
Returns a `request_id`.

2. **Check results**
```bash
GET /result/{request_id}
```

## 🛡️ Leakage & Corrosion Database

The system identifies multiple categories of infrastructure issues:
- **Intact Structure**: No issues detected.
- **Oil Leakage**: Active discharge detected.
- **Surface Corrosion**: Early stages of material degradation.
- **Severe Corrosion**: Critical structural damage.

## 🔄 Pipeline Management

The project uses **DVC** to manage the lifecycle of the model and data.

```bash
dvc repro
```

## 📈 Experiment Tracking

All training runs are logged to **MLflow**, tracking:
- Accuracy and Recall metrics.
- Model hyperparameters.
- Artifacts and serialized models.

---

**Empowering industrial safety through automated vision and MLOps.**
