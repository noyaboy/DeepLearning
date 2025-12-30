# Lab 4: Model Pruning and Quantization

Optimize neural networks through pruning and quantization techniques.

## Objectives

- Understand model compression techniques
- Implement structured and unstructured pruning
- Apply quantization for model size reduction

## Tasks

### Task 1: Model Pruning (`314580042_task1_pruning.ipynb`)

Reduce model size by removing less important weights.

- **Unstructured Pruning**: Remove individual weights based on magnitude
- **Structured Pruning**: Remove entire channels/filters
- **Iterative Pruning**: Gradually increase sparsity during training

### Task 2: Quantization

#### Part 1: Quantization API (`314580042_task2_1_quantization_api.ipynb`)

Use PyTorch's built-in quantization API.

- Post-training static quantization
- Quantization-aware training (QAT)
- INT8 inference

#### Part 2: Manual Quantization (`314580042_task2_2_quantization_manual.ipynb`)

Implement quantization from scratch.

- Symmetric and asymmetric quantization
- Scale and zero-point computation
- Quantized operations

## Results

| Technique | Model Size Reduction | Accuracy Impact |
|-----------|---------------------|-----------------|
| Pruning | See report | See report |
| INT8 Quantization | ~4x | Minimal |

## Files

| File | Description |
|------|-------------|
| `314580042_task1_pruning.ipynb` | Pruning implementation |
| `314580042_task2_1_quantization_api.ipynb` | PyTorch quantization API |
| `314580042_task2_2_quantization_manual.ipynb` | Manual quantization |
| `314580042_report.pdf` | Lab report |
| `2025_DL_Lab04.pdf` | Lab specification |
