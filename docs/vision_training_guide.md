# Vision Classifier Training Guide

## Overview
This guide covers the YOLOv8-based palm disease classifier that follows a "Green AI" approach - using efficient models and optimized formats for minimal environmental impact.

## Quick Start

### 1. Install Dependencies
```bash
# Activate virtual environment
& .\.venv\bin\Activate.ps1

# Install new dependencies
pip install -r requirements.txt
```

This will install:
- `ultralytics` - YOLOv8 framework
- `torch` & `torchvision` - Deep learning backend
- `onnx` & `onnxruntime` - Model export and inference

### 2. Run Training

**Standalone:**
```bash
python src/vision/train_classifier.py
```

**Via DVC Pipeline:**
```bash
# Run full pipeline including data organization
dvc repro

# Or just the vision training stage
dvc repro train_vision_model
```

## Green AI Features

### What Makes This "Green AI"?

1. **Nano Model** - Uses YOLOv8n (nano), the smallest YOLO variant
2. **ONNX Export** - Converts to optimized runtime format (~3-5MB vs 20+ MB PyTorch)
3. **Efficient Training** - Only 20 epochs with 224x224 images
4. **Metrics Tracked** - Model size and training time logged to MLflow

### Expected Resource Usage

| Metric | Value |
|--------|-------|
| Training Time | ~5-10 minutes (CPU), ~2-3 minutes (GPU) |
| Final Model Size | 3-5 MB (ONNX format) |
| Disk Space (training) | ~100-200 MB |
| RAM Required | ~2-4 GB |

## Configuration

### Default Settings
```python
DATA_PATH = "data/processed/palm_disease_final"
MODEL_SAVE_PATH = "models/palm_classifier.onnx"
EPOCHS = 20
IMAGE_SIZE = 224
BATCH_SIZE = 16
```

### Customization
You can modify these in `src/vision/train_classifier.py`:

```python
results = model.train(
    data=DATA_PATH,
    epochs=50,           # Increase for better accuracy
    imgsz=640,          # Larger images = better but slower
    batch=32,           # Increase if you have more RAM/VRAM
)
```

## Outputs

### Model Files
- `models/palm_classifier.onnx` - Final exported model (tracked by DVC)
- `palm_project/green_experiment/weights/best.pt` - Best PyTorch checkpoint

### MLflow Logs
View in MLflow UI:
```bash
mlflow ui
# Visit: http://localhost:5000
```

**Logged Metrics:**
- `training_duration_seconds` - Total training time
- `model_size_mb` - Final ONNX model size
- Model artifact (ONNX file)

## Pipeline Integration

### DVC Pipeline Flow
```
organize_palm_data (data reorganization)
    ↓
train_vision_model (YOLOv8 training)
    ↓
models/palm_classifier.onnx
```

The vision training automatically runs after data organization and will re-train if:
- The source code (`train_classifier.py`) changes
- The input data (`palm_disease_final/`) changes

### Parallel Pipelines
Note that `run_pipeline` (scikit-learn crop recommendation) and `train_vision_model` (YOLOv8 disease classification) are **independent** pipelines that can run in parallel:

- `run_pipeline` → Crop yield prediction (tabular data)
- `train_vision_model` → Disease classification (image data)

## Disease Classes

The model classifies 9 palm disease categories:
1. Black Scorch
2. Fusarium Wilt
3. Healthy Sample
4. Leaf Spots
5. Magnesium Deficiency
6. Manganese Deficiency
7. Parlatoria Blanchardi
8. Potassium Deficiency
9. Rachis Blight

## Troubleshooting

### "No module named 'ultralytics'"
```bash
pip install -r requirements.txt
```

### "CUDA out of memory"
Reduce batch size in `train_classifier.py`:
```python
batch=8,  # or even 4
```

### "FileNotFoundError: data/processed/palm_disease_final"
Run data organization first:
```bash
python src/data/organise_data.py
# OR
dvc repro organize_palm_data
```

### Training is very slow
- **Use GPU**: Install PyTorch with CUDA support
- **Reduce epochs**: Change `epochs=20` to `epochs=10`
- **Smaller images**: Change `imgsz=224` to `imgsz=128`

## Next Steps

After training:
1. **Evaluate the model** - Check accuracy metrics in MLflow
2. **Test inference** - Load ONNX model and test on new images
3. **Deploy** - Use ONNX model in production for fast inference
4. **Iterate** - Adjust hyperparameters based on results

## Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Green AI Best Practices](https://github.com/Green-Software-Foundation)
