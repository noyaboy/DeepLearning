# Lab 1: Custom Neural Network Framework

Build a neural network framework from scratch using NumPy.

## Objectives

- Implement forward and backward propagation manually
- Understand gradient computation and optimization
- Build custom layers without deep learning frameworks

## Implementation

### Custom Layers (`layer_314580042.py`)

| Layer | Description |
|-------|-------------|
| `Convolution` | 2D convolution with im2col optimization |
| `BatchNorm` | Batch normalization with moving statistics |
| `SiLU` | SiLU/Swish activation function |
| `FullyConnected` | Dense layer |
| `GlobalAveragePooling` | Channel-wise average pooling |
| `SoftmaxWithLoss` | Softmax with cross-entropy loss and label smoothing |
| `Block` | Residual block with dual pathways |

### Network Architecture (`network_314580042.py`)

- Input: 28x28 images (reshaped to 1x28x28)
- Stem convolution layer
- 3 Residual blocks (64 -> 128 -> 256 channels)
- Global Average Pooling
- Fully Connected: 256 -> 10 classes

### Training Features

- Momentum optimizer with weight decay
- Random augmentations (horizontal flip, random crop)
- Exponential moving average for BatchNorm

## Files

| File | Description |
|------|-------------|
| `layer_314580042.py` | Custom layer implementations |
| `network_314580042.py` | Network architecture |
| `Lab01_task1_314580042.ipynb` | Task 1 notebook |
| `Lab01_task2_314580042.ipynb` | Task 2 notebook |
| `Lab01_report_314580042.pdf` | Lab report |
| `2025_DL_Lab01.pdf` | Lab specification |
