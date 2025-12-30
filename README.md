# Deep Learning Course

陽明交通大學 IEE535223 電子碩 張添烜教授
This repository contains implementations, reports, and materials for a Deep Learning course (Labs 1-5) and a final team project.

## Repository Structure

```
.
├── lab1/           # Custom Neural Network Framework (NumPy)
├── lab2/           # Compact DenseNet (PyTorch)
├── lab3/           # Sequence-to-Sequence Translation (Transformer)
├── lab4/           # Model Pruning and Quantization
├── lab5/           # Semantic Segmentation with Optimization
├── final_project/  # Team14 - HDC-CNN with K-Fold Cross Validation
└── slides/         # Course Lecture Slides
```

## Lab Summaries

### Lab 1: Custom Neural Network Framework
Build a neural network framework from scratch using NumPy.
- Custom layers: Convolution, BatchNorm, SiLU, FullyConnected
- Residual blocks with skip connections
- Training with momentum optimizer and weight decay

**Files:** `layer_314580042.py`, `network_314580042.py`, notebooks, report

### Lab 2: Compact DenseNet
Implement a Compact DenseNet architecture in PyTorch.
- Local Dense Blocks (LDB) with skip connections
- Transition layers for channel reduction
- 212K parameters, 7.71 GigaFLOPs

**Files:** `CDenseNet_314580042.py`, notebook, report

### Lab 3: Sequence-to-Sequence Translation
Implement a Transformer model for Chinese-English translation.
- Multi-head attention mechanism
- Positional encoding
- Encoder-decoder architecture

**Files:** `network.py`, `utils.py`, `run.py`, notebook, report

### Lab 4: Model Pruning and Quantization
Optimize neural networks through pruning and quantization techniques.
- Task 1: Model pruning
- Task 2: Quantization (API and manual approaches)

**Files:** Notebooks for pruning and quantization, model checkpoints, report

### Lab 5: Semantic Segmentation
Implement and optimize U-Net for semantic segmentation.
- Task 1: Standard U-Net implementation
- Task 2: Efficient U-Net with depthwise separable convolutions, ECA attention, and INT8 quantization

**Files:** `network_task1.py`, `network_task2.py`, notebooks, report

### Final Project: HDC-CNN
Team14 project implementing HDC-CNN with k-fold cross validation.
- Multi-GPU training support
- Isotonic calibration
- Performance-weighted ensemble

**Files:** Complete project structure with models, data pipelines, configs, presentation, and video link

## Course Slides

| Lecture | Topic |
|---------|-------|
| 0 | Course Introduction |
| 1 | Overview of Computer Vision on Image Classification |
| 2 | Neural Network |
| 3 | CNN |
| 4-1 | Training a Neural Network |
| 4-2 | Better Accuracy |
| 4-3 | Gradient Descent Optimization |
| 4-4 | Regularization and Augmentation |
| 4-5 | Nuts and Bolts of Deep Learning |
| 5 | Advanced CNN Architectures |
| 6 | RNN |
| 6-2 | Transformer |
| 6-3 | Vision Transformer |
| 7-1 | Pruning |
| 7-2 | Quantization |
| 7-3 | Tensor Decomposition |
| 7-4 | Low Complexity Model |
| 8-1 | Semantic and Instance Segmentation |
| 8-2 | Object Detection |
| 8-3 | Object Detection - New Trends |
| 9 | GAN |
| 9-2 | Diffusion |
| 10 | Pretrained Model |
| 10-2 | Self-Supervised Learning |

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- Jupyter Notebook

## Author

Student ID: 314580042
