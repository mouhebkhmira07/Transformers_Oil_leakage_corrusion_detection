# Organize Data Script Usage

## Overview
The `organise_data.py` script reorganizes and splits image datasets into train/validation/test sets for machine learning pipelines.

## Location
`src/data/organise_data.py`

## Features
- ✅ Automatically cleans folder names (removes numbering, normalizes formatting)
- ✅ Splits data into train/val/test sets with configurable ratios
- ✅ Supports command-line arguments for flexibility
- ✅ Integrated with DVC pipeline
- ✅ Integrated with GitHub Actions workflow

## Usage

### Basic Usage (Default Settings)
```bash
python src/data/organise_data.py
```

**Defaults:**
- Source: `data/Infected Date Palm Leaves Dataset/Processed`
- Destination: `data/processed/palm_disease_final`
- Train ratio: 70%
- Validation ratio: 20%
- Test ratio: 10%

### Custom Source and Destination
```bash
python src/data/organise_data.py --source "path/to/source" --dest "path/to/destination"
```

### Custom Split Ratios
```bash
python src/data/organise_data.py --train-ratio 0.8 --val-ratio 0.15 --test-ratio 0.05
```

### Full Example with All Arguments
```bash
python src/data/organise_data.py \
  --source "data/Infected Date Palm Leaves Dataset/Processed" \
  --dest "data/processed/palm_disease_custom" \
  --train-ratio 0.75 \
  --val-ratio 0.15 \
  --test-ratio 0.10
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | string | `data/Infected Date Palm Leaves Dataset/Processed` | Source path with class folders |
| `--dest` | string | `data/processed/palm_disease_final` | Destination for organized dataset |
| `--train-ratio` | float | `0.7` | Training set proportion |
| `--val-ratio` | float | `0.2` | Validation set proportion |
| `--test-ratio` | float | `0.1` | Test set proportion |

## Pipeline Integration

### DVC Pipeline
The script is integrated into the DVC pipeline as the `organize_palm_data` stage:

```bash
# Run only the data organization stage
dvc repro organize_palm_data

# Run the full pipeline (including data organization)
dvc repro
```

### GitHub Actions
The script automatically runs in the CI/CD pipeline before the main processing pipeline.

## Output Structure
The script creates the following directory structure:

```
data/processed/palm_disease_final/
├── train/
│   ├── class_name_1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class_name_2/
│   └── ...
├── val/
│   ├── class_name_1/
│   ├── class_name_2/
│   └── ...
└── test/
    ├── class_name_1/
    ├── class_name_2/
    └── ...
```

## Expected Input Structure
The source directory should contain class folders with images:

```
data/Infected Date Palm Leaves Dataset/Processed/
├── 1. Potassium Deficiency/
│   ├── img001.jpg
│   ├── img002.jpg
│   └── ...
├── 2. Black Scorch/
│   ├── img001.jpg
│   └── ...
└── ...
```

## Notes
- ✓ Folder names are automatically cleaned (e.g., "1. Potassium Deficiency" → "potassium_deficiency")
- ✓ Images are randomly shuffled before splitting
- ✓ Supports JPG, JPEG, and PNG formats
- ✓ Creates directories automatically if they don't exist
- ✓ Warns if destination already exists (merges/overwrites)
- ✓ Skips folders with no valid images

## Troubleshooting

### "Path does not exist" Error
Make sure the source path exists and contains folders with images.

### "No folders found" Error
Verify that the source path contains subdirectories (class folders) with images.

### Empty Output
Check that image files have valid extensions (.jpg, .jpeg, .png) and are not corrupted.
