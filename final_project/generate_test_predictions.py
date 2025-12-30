# Generate test predictions from K-fold trained HDC-CNN models.

import sys
import argparse
from pathlib import Path
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset import CosmologyDataset
from models.registry import ModelRegistry
from models.hdc_cnn import HDCCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate test predictions from K-fold models'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Run name (e.g., hdc_cnn_kfold_a100_stable)'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=4,
        help='Number of folds (default: 4)'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=2,
        help='Number of repeats (default: 2)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='../public_data/',
        help='Directory containing test data'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )

    return parser.parse_args()


def load_variant_selection(val_summary_dir: Path, fold_id: int, repeat_id: int):
    variant_file = val_summary_dir / f'f{fold_id}r{repeat_id}_variant_selection.json'

    with open(variant_file) as f:
        data = json.load(f)

    return data['best_variant'], data['best_score']


def load_and_preprocess_test_data(data_dir: str, image_mean: float, image_std: float):
    import gc

    print("\nLoading test data...")
    mask_file = Path(data_dir) / 'WIDE12H_bin2_2arcmin_mask.npy'
    mask = np.load(mask_file)

    test_file = Path(data_dir) / 'WIDE12H_bin2_2arcmin_kappa_noisy_test.npy'
    test_data_masked = np.load(test_file).astype(np.float32)
    print(f"  Loaded: {test_data_masked.shape}")

    n_samples = test_data_masked.shape[0]
    H, W = 1424, 176

    test_data = np.zeros((n_samples, H, W), dtype=np.float32)
    test_data[:, mask] = test_data_masked

    del test_data_masked
    gc.collect()

    test_data -= image_mean
    test_data /= image_std

    print(f"  Normalized {n_samples} samples (mean={image_mean:.6f}, std={image_std:.6f})")

    test_dataset = CosmologyDataset(test_data, labels=None)

    return test_dataset, n_samples


def run_inference(model, test_loader, device='cuda'):
    model.eval()

    predictions_mean = []
    predictions_std = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Inference", leave=False):
            images = batch['image'].to(device)

            outputs = model({'image': images})

            mean = outputs['mean'].cpu().numpy()
            log_var = outputs['log_var'].cpu().numpy()

            std = np.exp(0.5 * log_var)

            predictions_mean.append(mean)
            predictions_std.append(std)

    predictions_mean = np.concatenate(predictions_mean, axis=0)
    predictions_std = np.concatenate(predictions_std, axis=0)

    return predictions_mean, predictions_std


def denormalize_predictions(predictions_mean, predictions_std, label_mean, label_std):
    means_denorm = predictions_mean * label_std + label_mean
    stds_denorm = predictions_std * label_std

    return means_denorm, stds_denorm


def main():
    args = parse_args()
    start_time = time.time()

    print(f"\nGenerating test predictions")
    print(f"Run: {args.run_name}, Folds: {args.n_folds}, Repeats: {args.n_repeats}")

    checkpoint_dir = Path(f'checkpoints/{args.run_name}')
    output_dir = Path(f'outputs/{args.run_name}')
    val_summary_dir = output_dir / 'val_summary'
    test_pred_dir = output_dir / 'test_predictions'
    test_pred_dir.mkdir(parents=True, exist_ok=True)

    global_norm_path = output_dir / 'global_normalization.json'
    with open(global_norm_path) as f:
        global_norm = json.load(f)

    image_mean = global_norm['image_mean']
    image_std = global_norm['image_std']

    print(f"\nGlobal normalization:")
    print(f"  image_mean: {image_mean}")
    print(f"  image_std: {image_std}")

    test_dataset, n_samples = load_and_preprocess_test_data(
        args.data_dir, image_mean, image_std
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    print("\nRunning inference per model...")

    for repeat_id in range(args.n_repeats):
        for fold_id in range(args.n_folds):
            print(f"\nFold {fold_id}, Repeat {repeat_id}")

            best_variant, best_score = load_variant_selection(
                val_summary_dir, fold_id, repeat_id
            )
            print(f"  Best variant: {best_variant} (score: {best_score:.4f})")

            fold_dir = checkpoint_dir / f'fold{fold_id}_rep{repeat_id}'
            checkpoint_path = fold_dir / 'best_model.pth'

            print(f"  Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            label_mean = checkpoint['label_scaler']['mean']
            label_std = checkpoint['label_scaler']['scale']

            model = ModelRegistry.build('hdc_cnn', {})

            if best_variant == 'ema':
                model.load_state_dict(checkpoint['ema_state_dict']['shadow'], strict=False)
            else:
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            model.to(device)

            print(f"  Running inference on {n_samples} test samples...")
            pred_mean, pred_std = run_inference(model, test_loader, device)

            means_denorm, stds_denorm = denormalize_predictions(
                pred_mean, pred_std, label_mean, label_std
            )

            df = pd.DataFrame({
                'sample_id': np.arange(n_samples),
                'fold_id': fold_id,
                'repeat_id': repeat_id,
                'mu_Om': means_denorm[:, 0],
                'mu_S8': means_denorm[:, 1],
                'var_Om': stds_denorm[:, 0] ** 2,
                'var_S8': stds_denorm[:, 1] ** 2,
                'model_id': f'fold{fold_id}_rep{repeat_id}',
                'ckpt_type': best_variant,
                'val_score': best_score
            })

            output_file_parquet = test_pred_dir / f'test_fold{fold_id}rep{repeat_id}_{best_variant}.parquet'
            output_file_pkl = test_pred_dir / f'test_fold{fold_id}rep{repeat_id}_{best_variant}.pkl'
            try:
                df.to_parquet(output_file_parquet, index=False)
                output_file = output_file_parquet
            except ImportError:
                df.to_pickle(output_file_pkl)
                output_file = output_file_pkl

            print(f"  Saved: {output_file}")
            print(f"  Om: [{means_denorm[:, 0].min():.4f}, {means_denorm[:, 0].max():.4f}], "
                  f"S8: [{means_denorm[:, 1].min():.4f}, {means_denorm[:, 1].max():.4f}]")

            del model, checkpoint
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    elapsed_time = time.time() - start_time

    print(f"\nDone: {args.n_folds * args.n_repeats} files, {n_samples} samples each")
    print(f"Output: {test_pred_dir}")
    print(f"Time: {elapsed_time/60:.1f} min")


if __name__ == '__main__':
    main()
