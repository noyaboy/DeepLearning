# Final Project: HDC-CNN for HMS - Harmful Brain Activity Classification

## Team 14

### Presentation

- [Presentation Slides (PDF)](DL_FP_Team14.pdf)
- [Presentation Slides (PPTX)](DL_FP_Team14.pptx)

### Video Presentation

[![Video Presentation](https://img.youtube.com/vi/otA_Glv_Dfk/0.jpg)](https://www.youtube.com/watch?v=otA_Glv_Dfk)

[Watch on YouTube](https://www.youtube.com/watch?v=otA_Glv_Dfk)

## Project Overview

HDC-CNN (Hierarchical Deep CNN) implementation for the HMS - Harmful Brain Activity Classification Kaggle competition.

### Key Features

- K-fold cross validation with multiple repetitions
- Multi-GPU training support
- Isotonic calibration for probability calibration
- Performance-weighted ensemble

### Project Structure

```
final_project/
├── configs/                  # Training configurations
├── checkpoints/              # Model checkpoints (not tracked)
├── data/                     # Data loading and preprocessing
│   ├── augmentations.py
│   ├── dataset.py
│   ├── loader.py
│   ├── preprocessing.py
│   └── splits.py
├── evaluation/               # Evaluation metrics
│   └── scoring.py
├── experiments/              # Training scripts
│   ├── train_kfold.py
│   ├── train_kfold_multigpu.py
│   └── train_single_fold_worker.py
├── models/                   # Model architectures
│   ├── base.py
│   ├── hdc_cnn.py
│   ├── layers.py
│   └── registry.py
├── apply_isotonic_calibration.py
├── create_final_calibrated_submission.py
├── create_performance_weighted_ensemble.py
├── generate_oof_predictions.py
└── generate_test_predictions.py
```

### Usage

1. Prepare data in parquet format
2. Configure training in `configs/`
3. Run k-fold training:
   ```bash
   python experiments/train_kfold.py --config configs/hdc_cnn_kfold_a100_stable.yaml
   ```
4. Generate predictions and create ensemble:
   ```bash
   # Run inference pipeline
   python generate_test_predictions.py --run-name hdc_cnn_kfold_a100_stable && \
   python create_performance_weighted_ensemble.py --run-name hdc_cnn_kfold_a100_stable && \
   python run_pit_diagnostics.py --run-name hdc_cnn_kfold_a100_stable && \
   python apply_isotonic_calibration.py --run-name hdc_cnn_kfold_a100_stable && \
   python create_final_calibrated_submission.py --run-name hdc_cnn_kfold_a100_stable
   ```
