# Lab 5: Semantic Segmentation with Optimization

Implement and optimize U-Net for semantic segmentation.

## Objectives

- Implement U-Net architecture for semantic segmentation
- Apply model optimization techniques
- Achieve efficient inference with quantization

## Tasks

### Task 1: Standard U-Net (`network_task1.py`)

Basic U-Net implementation for 8-class segmentation.

#### Architecture
- **Encoder**: 4 down-sampling blocks with DoubleConv + MaxPool
- **Decoder**: 4 up-sampling blocks with skip connections
- **Output**: 8 classes

```
Input -> Encoder -> Bottleneck -> Decoder -> Output
         |_________Skip Connections_________|
```

### Task 2: Efficient U-Net (`network_task2.py`)

Optimized U-Net with quantization support.

#### Optimizations

| Technique | Description |
|-----------|-------------|
| Depthwise Separable Conv | Reduces parameters and computation |
| ECA Attention | Efficient Channel Attention mechanism |
| Bilinear Upsampling | Replaces transposed convolution |
| INT8 Quantization | 4x size reduction with QAT |

#### Quantization Features

- `QuantStub/DeQuantStub` for QAT pipeline
- Support for qnnpack backend
- `QuantizedModelWrapper` for CPU inference

## Files

| File | Description |
|------|-------------|
| `network_task1.py` | Standard U-Net |
| `network_task2.py` | Efficient U-Net with quantization |
| `Lab05_task1_314580042.ipynb` | Task 1 notebook |
| `Lab05_task2_314580042.ipynb` | Task 2 notebook |
| `report_314580042.pdf` | Lab report |
| `2025_DL_Lab05_V2.pdf` | Lab specification |

## Usage

```python
from network_task2 import EfficientUNet, load_model

# Load FP32 or INT8 model
model = load_model('model_path')

# For quantization-aware training
from network_task2 import prepare_qat_model, convert_to_quantized
model = prepare_qat_model(model)
# ... train ...
quantized_model = convert_to_quantized(model)
```
