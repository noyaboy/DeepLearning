# Lab 2: Compact DenseNet

Implement a Compact DenseNet architecture in PyTorch for image classification.

## Objectives

- Understand DenseNet architecture and dense connections
- Implement Local Dense Blocks (LDB)
- Optimize model for parameter efficiency

## Implementation

### Architecture (`CDenseNet_314580042.py`)

#### Local Dense Block (LDB)
- Initial convolution reduces channels by factor t (0.5)
- 4 sequential conv layers with skip connections
- Output concatenates all branches

#### Transition Layer
- 1x1 convolution to reduce channels to 32

#### CDenseNet
- Stem: Conv2d (1 -> 32 channels)
- 16 LDB blocks with transition layers
- Global Average Pooling
- Classifier: Linear(32 -> 128 -> 3)

### Model Statistics

| Metric | Value |
|--------|-------|
| Total Parameters | 212,835 |
| Model Size | 0.85 MB |
| GigaFLOPs | 7.71 |
| Input Size | [1, 1, 158, 238] |

### Weight Initialization

- Conv2d: Kaiming normal (fan_out, relu)
- BatchNorm: weight=1.0, bias=0.0
- Linear: Xavier uniform

## Files

| File | Description |
|------|-------------|
| `CDenseNet_314580042.py` | Model implementation |
| `Lab02_CDenseNet_314580042.ipynb` | Training notebook |
| `summary_314580042.txt` | Model architecture summary |
| `Lab02_report_314580042.pdf` | Lab report |
| `2025_DL_Lab02_v2.pdf` | Lab specification |
