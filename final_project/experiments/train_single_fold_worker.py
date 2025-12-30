#!/usr/bin/env python3
# Single-fold training worker for multi-GPU orchestration.

import argparse
from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_kfold import train_single_fold, Config, WLDataLoader
from experiments.train_kfold import compute_global_normalization, create_kfold_splits_by_systematics
from experiments.train_kfold import NOISE_BATCH_SIZE
from data.preprocessing import Preprocessor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--fold-id', type=int, required=True)
    parser.add_argument('--repeat-id', type=int, required=True)
    args = parser.parse_args()

    print(f"Fold {args.fold_id}, Repeat {args.repeat_id}")

    config = Config(args.config)

    data_loader = WLDataLoader(
        data_dir=config.get('data.data_dir'),
        use_public_dataset=config.get('data.use_public_dataset'),
        max_cosmologies=config.get('data.max_cosmologies', None)
    )
    print(f"Data: {data_loader.Ncosmo} x {data_loader.Nsys}")

    global_norm_path = Path(config.get('paths.output_dir')) / 'global_normalization.json'

    if global_norm_path.exists():
        with open(global_norm_path, 'r') as f:
            global_norm_stats = json.load(f)

        kappa_flat = data_loader.kappa.reshape(-1, data_loader.img_height, data_loader.img_width)
        kappa_noisy = np.zeros_like(kappa_flat)

        noise_seed = config.get('data.noise_seed')
        n = kappa_flat.shape[0]
        for i in range(0, n, NOISE_BATCH_SIZE):
            end = min(i + NOISE_BATCH_SIZE, n)
            kappa_noisy[i:end] = Preprocessor.add_noise(
                kappa_flat[i:end],
                np.broadcast_to(data_loader.mask, kappa_flat[i:end].shape),
                data_loader.pixel_size_arcmin,
                data_loader.galaxy_density_per_arcmin2,
                data_loader.shape_noise,
                seed=noise_seed + i
            )
        print(f"Noisy data: {n} samples")
    else:
        kappa_noisy, global_norm_stats = compute_global_normalization(
            data_loader,
            noise_seed=config.get('data.noise_seed'),
            output_path=global_norm_path
        )

    kfold_splits = create_kfold_splits_by_systematics(
        n_cosmologies=data_loader.Ncosmo,
        n_systematics=data_loader.Nsys,
        n_folds=config.get('data.n_folds'),
        seed=config.get('data.split_seed') + args.repeat_id
    )

    labels_flat = data_loader.labels[:, :, :2].reshape(-1, 2)
    train_idx, val_idx = kfold_splits[args.fold_id]
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    print("Starting training...")
    train_single_fold(
        fold_id=args.fold_id,
        repeat_id=args.repeat_id,
        train_idx=train_idx,
        val_idx=val_idx,
        kappa_noisy=kappa_noisy,
        labels_flat=labels_flat,
        global_norm_stats=global_norm_stats,
        config=config
    )

    print("Done")


if __name__ == '__main__':
    main()
