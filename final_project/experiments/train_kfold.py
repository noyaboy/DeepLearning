# K-Fold Cross-Validation Training Script for HDC-CNN.
# Implements global normalization, K-fold CV (5 folds x 2 repeats = 10 models),
# up to 40 epochs with early stopping, and model variant selection (RAW vs EMA).

import sys
import os
from pathlib import Path
import json
import argparse
import copy

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

NOISE_BATCH_SIZE = 500
from data.loader import DataLoader as WLDataLoader
from data.preprocessing import Preprocessor
from data.dataset import CosmologyDataset
from data.splits import create_kfold_splits_by_systematics, verify_split
from models.registry import ModelRegistry
from models.hdc_cnn import HDCCNN  # Import to register
from training.trainer import Trainer
from training.losses import get_loss_function
from training.optimizers import get_optimizer
from training.schedulers import get_scheduler
from training.callbacks.checkpoint import ModelCheckpoint
from training.callbacks.early_stopping import EarlyStopping
from training.callbacks.logging import LoggingCallback
from training.callbacks.oof_logger import OOFLogger
from training.callbacks.test_logger import TestLogger
from utils.config import Config
from utils.reproducibility import configure_reproducibility
from evaluation.scoring import CompetitionScorer


def compute_global_normalization(
    data_loader,
    noise_seed: int,
    output_path: Path
) -> dict:
    print("\nComputing global normalization...")
    kappa_flat = data_loader.kappa.reshape(-1, data_loader.img_height, data_loader.img_width)
    n_samples = kappa_flat.shape[0]

    count = 0
    mean = 0.0

    for i in range(0, n_samples, NOISE_BATCH_SIZE):
        end_idx = min(i + NOISE_BATCH_SIZE, n_samples)
        print(f"  mean {i}-{end_idx}/{n_samples}")

        kappa_noisy_batch = Preprocessor.add_noise(
            kappa=kappa_flat[i:end_idx],
            mask=data_loader.mask,
            pixel_size_arcmin=data_loader.pixel_size_arcmin,
            galaxy_density=data_loader.galaxy_density_per_arcmin2,
            shape_noise=data_loader.shape_noise,
            seed=noise_seed + i
        )

        batch_mean = np.mean(kappa_noisy_batch)
        batch_count = kappa_noisy_batch.size
        count += batch_count
        mean += (batch_mean - mean) * batch_count / count

        del kappa_noisy_batch

    global_mean = mean

    print("Computing std...")
    M2 = 0.0

    for i in range(0, n_samples, NOISE_BATCH_SIZE):
        end_idx = min(i + NOISE_BATCH_SIZE, n_samples)
        print(f"  std {i}-{end_idx}/{n_samples}")

        kappa_noisy_batch = Preprocessor.add_noise(
            kappa=kappa_flat[i:end_idx],
            mask=data_loader.mask,
            pixel_size_arcmin=data_loader.pixel_size_arcmin,
            galaxy_density=data_loader.galaxy_density_per_arcmin2,
            shape_noise=data_loader.shape_noise,
            seed=noise_seed + i
        )

        M2 += np.sum((kappa_noisy_batch - global_mean) ** 2)

        del kappa_noisy_batch

    global_std = np.sqrt(M2 / count) if count > 1 else 0.0

    print("Generating noisy data...")
    kappa_noisy = np.empty_like(kappa_flat)

    for i in range(0, n_samples, NOISE_BATCH_SIZE):
        end_idx = min(i + NOISE_BATCH_SIZE, n_samples)
        print(f"  gen {i}-{end_idx}/{n_samples}")

        kappa_noisy[i:end_idx] = Preprocessor.add_noise(
            kappa=kappa_flat[i:end_idx],
            mask=data_loader.mask,
            pixel_size_arcmin=data_loader.pixel_size_arcmin,
            galaxy_density=data_loader.galaxy_density_per_arcmin2,
            shape_noise=data_loader.shape_noise,
            seed=noise_seed + i
        )

    normalization_stats = {
        'image_mean': float(global_mean),
        'image_std': float(global_std),
        'noise_seed': int(noise_seed),
        'n_samples': int(n_samples),
        'noise_batch_size': int(NOISE_BATCH_SIZE)
    }

    if abs(global_mean) > 0.01:
        print(f"  global_mean={global_mean:.6f} (larger than expected)")

    if global_std < 0.001 or global_std > 1.0:
        print(f"  global_std={global_std:.6f} (outside [0.001, 1.0])")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(normalization_stats, f, indent=2)

    print(f"Normalization: mean={global_mean:.6f}, std={global_std:.6f}")
    print(f"Saved: {output_path}")

    return kappa_noisy, normalization_stats


def select_best_variant(
    trainer,
    val_loader,
    label_scaler,
    fold_id: int,
    repeat_id: int,
    device: str
) -> tuple:
    print(f"\nVariant selection (fold={fold_id}, repeat={repeat_id})")

    scorer = CompetitionScorer(label_scaler=label_scaler)
    variants = {}
    scores = {}

    print("\nEvaluating RAW variant...")
    trainer.model.eval()
    raw_score = scorer.compute_score(trainer.model, val_loader, device)
    variants['raw'] = copy.deepcopy(trainer.model.state_dict())
    scores['raw'] = raw_score
    print(f"  RAW score: {raw_score:.6f}")

    if trainer.ema is not None:
        print("\nEvaluating EMA variant...")
        with trainer.use_ema_for_evaluation():
            ema_score = scorer.compute_score(trainer.model, val_loader, device)
            variants['ema'] = copy.deepcopy(trainer.model.state_dict())
            scores['ema'] = ema_score
        print(f"  EMA score: {ema_score:.6f}")

    best_variant = max(scores, key=scores.get)
    best_score = scores[best_variant]

    print(f"Best: {best_variant} (score={best_score:.6f})")

    return best_variant, best_score, scores, variants


def train_single_fold(
    fold_id: int,
    repeat_id: int,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    kappa_noisy: np.ndarray,
    labels_flat: np.ndarray,
    global_norm_stats: dict,
    config: Config,
    test_loader=None
):
    print(f"\n=== Fold {fold_id+1}/{config.get('data.n_folds')}, Rep {repeat_id+1}/{config.get('data.n_repeats')} ===")

    fold_seed = config.get('reproducibility.seed') + repeat_id * 10000 + fold_id
    configure_reproducibility(fold_seed, config.get('reproducibility.deterministic'),
                              config.get('reproducibility.benchmark'))

    fold_output_dir = Path(config.get('paths.output_dir')) / f'fold{fold_id}_rep{repeat_id}'
    fold_checkpoint_dir = Path(config.get('paths.checkpoint_dir')) / f'fold{fold_id}_rep{repeat_id}'
    fold_oof_dir = Path(config.get('paths.oof_dir'))
    fold_test_dir = Path(config.get('paths.test_pred_dir'))
    fold_val_summary_dir = Path(config.get('paths.val_summary_dir'))

    for d in [fold_output_dir, fold_checkpoint_dir, fold_oof_dir, fold_test_dir, fold_val_summary_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print(f"Data: {len(train_idx)} train, {len(val_idx)} val")

    import tempfile
    import gc
    batch_size, sub_batch_size, chunk_size = 1000, 100, 1000

    # Extract training data to memmap
    temp_train = tempfile.NamedTemporaryFile(delete=False, suffix='_train.dat')
    temp_train.close()
    X_train = np.memmap(temp_train.name, dtype='float32', mode='w+',
                        shape=(len(train_idx), kappa_noisy.shape[1], kappa_noisy.shape[2]))

    for i in range(0, len(train_idx), batch_size):
        end = min(i + batch_size, len(train_idx))
        for j in range(0, end - i, sub_batch_size):
            X_train[i+j:i+min(j+sub_batch_size, end-i)] = kappa_noisy[train_idx[i+j:i+min(j+sub_batch_size, end-i)]]
        if (i // batch_size) % 10 == 0:
            X_train.flush()
    X_train.flush()

    # Extract validation data to memmap
    temp_val = tempfile.NamedTemporaryFile(delete=False, suffix='_val.dat')
    temp_val.close()
    X_val = np.memmap(temp_val.name, dtype='float32', mode='w+',
                      shape=(len(val_idx), kappa_noisy.shape[1], kappa_noisy.shape[2]))

    for i in range(0, len(val_idx), batch_size):
        end = min(i + batch_size, len(val_idx))
        for j in range(0, end - i, sub_batch_size):
            X_val[i+j:i+min(j+sub_batch_size, end-i)] = kappa_noisy[val_idx[i+j:i+min(j+sub_batch_size, end-i)]]
    X_val.flush()

    y_train = labels_flat[train_idx].copy()
    y_val = labels_flat[val_idx].copy()

    if test_loader is None:
        del kappa_noisy, labels_flat
        gc.collect()

    memmap_files = [temp_train.name, temp_val.name]

    # Apply normalization
    preprocessor = Preprocessor()
    preprocessor.image_mean = global_norm_stats['image_mean']
    preprocessor.image_std = global_norm_stats['image_std']
    X_train = preprocessor.transform_images(X_train, inplace=True)
    X_val = preprocessor.transform_images(X_val, inplace=True)

    # Copy to RAM for faster DataLoader access
    X_train_ram = np.empty(X_train.shape, dtype=X_train.dtype)
    for i in range(0, len(X_train), chunk_size):
        X_train_ram[i:i+chunk_size] = X_train[i:i+chunk_size]
    del X_train
    gc.collect()
    X_train = X_train_ram

    X_val_ram = np.empty(X_val.shape, dtype=X_val.dtype)
    for i in range(0, len(X_val), chunk_size):
        X_val_ram[i:i+chunk_size] = X_val[i:i+chunk_size]
    del X_val
    gc.collect()
    X_val = X_val_ram

    y_train = preprocessor.fit_transform_labels(y_train)
    y_val = preprocessor.transform_labels(y_val)

    if len(val_idx) == 0:
        raise ValueError(f"Empty validation set for fold {fold_id}")

    from data.augmentations import get_train_transforms, get_val_transforms

    train_dataset = CosmologyDataset(X_train, y_train, get_train_transforms(), train_idx)
    val_dataset = CosmologyDataset(X_val, y_val, get_val_transforms(), val_idx)

    loader_kwargs = dict(batch_size=config.get('training.batch_size'), num_workers=8,
                         pin_memory=True, persistent_workers=True, prefetch_factor=2)
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    model = ModelRegistry.build(config.get('model.name'), config.get('model'))
    print(f"Model: {config.get('model.name')} ({model.count_parameters():,} params)")

    if config.get('training.use_compile', False) and hasattr(torch, 'compile'):
        model = torch.compile(model, mode=config.get('training.compile_mode', 'reduce-overhead'),
                              fullgraph=False, dynamic=False)

    device = config.get('device')
    loss_fn = get_loss_function(config.get('training.loss'), **config.get('training.loss_params'))
    optimizer = get_optimizer(model, config.get('training.optimizer'))

    steps_per_epoch = len(train_loader)
    scheduler = get_scheduler(optimizer, config.get('training.scheduler'), steps_per_epoch=steps_per_epoch)

    image_scaler = {
        'mean': float(preprocessor.image_mean),
        'std': float(preprocessor.image_std)
    }
    label_scaler = {
        'mean': preprocessor.label_scaler.mean_.tolist(),
        'scale': preprocessor.label_scaler.scale_.tolist()
    }

    beta_scheduler = None
    if config.get('training.beta_schedule.enabled'):
        from training.schedulers import BetaScheduler
        beta_scheduler = BetaScheduler(
            loss_fn=loss_fn,
            warmup_epochs=config.get('training.beta_schedule.warmup_epochs'),
            rampup_epochs=config.get('training.beta_schedule.rampup_epochs'),
            target_beta=config.get('training.beta_schedule.target_beta')
        )

    lambda_train_scheduler = None
    if config.get('training.lambda_train_schedule.enabled'):
        from training.schedulers import LambdaTrainScheduler
        lambda_train_scheduler = LambdaTrainScheduler(
            loss_fn=loss_fn,
            start_epoch=config.get('training.lambda_train_schedule.start_epoch'),
            rampup_epochs=config.get('training.lambda_train_schedule.rampup_epochs'),
            target_lambda=config.get('training.lambda_train_schedule.target_lambda')
        )

    callbacks = [
        ModelCheckpoint(
            checkpoint_dir=str(fold_checkpoint_dir),
            monitor=config.get('training.callbacks.checkpoint.monitor'),
            mode=config.get('training.callbacks.checkpoint.mode'),
            save_scalers=True,
            image_scaler=image_scaler,
            label_scaler=label_scaler,
            fold_id=fold_id,
            repeat_id=repeat_id,
            seed=fold_seed
        ),
        LoggingCallback(log_every=1)
    ]

    if config.get('training.callbacks.early_stopping.enabled'):
        callbacks.append(EarlyStopping(
            config.get('training.callbacks.early_stopping.patience'),
            config.get('training.callbacks.early_stopping.monitor'),
            config.get('training.callbacks.early_stopping.mode')
        ))

    warmstart_config = {
        'enabled': config.get('training.warmstart.enabled', False),
        'epochs': config.get('training.warmstart.epochs', 3)
    }

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        callbacks=callbacks,
        device=device,
        use_amp=config.get('training.use_amp'),
        grad_clip_max_norm=config.get('training.grad_clip_max_norm'),
        use_checkpointing=config.get('training.use_checkpointing'),
        label_scaler=preprocessor.label_scaler,
        use_ema=config.get('training.use_ema'),
        ema_decay=config.get('training.ema_decay'),
        use_sma=config.get('training.use_sma', False),
        sma_n_average=config.get('training.sma_n_average', 10),
        beta_scheduler=beta_scheduler,
        lambda_train_scheduler=lambda_train_scheduler,
        warmstart_config=warmstart_config
    )

    trainer.fit(config.get('training.epochs'))

    # Load best checkpoint
    checkpoint_path = fold_checkpoint_dir / 'best_model.pth'
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'ema_state_dict' in checkpoint and trainer.ema is not None:
            trainer.ema.load_state_dict(checkpoint['ema_state_dict'])
        if 'sma_state_dict' in checkpoint and trainer.sma is not None:
            trainer.sma.load_state_dict(checkpoint['sma_state_dict'])
        model.to(device)
    model.eval()

    best_variant, best_score, scores, variants = select_best_variant(
        trainer, val_loader, preprocessor.label_scaler, fold_id, repeat_id, device
    )

    variant_file = fold_val_summary_dir / f'f{fold_id}r{repeat_id}_variant_selection.json'
    with open(variant_file, 'w') as f:
        json.dump({'fold_id': fold_id, 'repeat_id': repeat_id, 'best_variant': best_variant,
                   'best_score': float(best_score), 'scores': {k: float(v) for k, v in scores.items()}}, f, indent=2)

    model.load_state_dict(variants[best_variant])
    model.to(device)
    model.eval()

    OOFLogger(val_loader, val_dataset, preprocessor.label_scaler,
              fold_id, repeat_id, str(fold_oof_dir), best_variant).on_train_end(trainer)

    if test_loader is not None:
        TestLogger(test_loader, preprocessor.label_scaler, fold_id, repeat_id,
                   str(fold_test_dir), best_variant).on_train_end(trainer)

    # Cleanup memmap files
    import os
    for f in memmap_files:
        try:
            os.unlink(f)
        except OSError:
            pass

    print(f"Fold {fold_id+1} done: {best_variant} score={best_score:.4f}")

    return {
        'fold_id': fold_id,
        'repeat_id': repeat_id,
        'best_variant': best_variant,
        'best_score': best_score,
        'scores': scores
    }


def main():
    parser = argparse.ArgumentParser(description='HDC-CNN K-Fold Cross-Validation Training')
    parser.add_argument('--config', type=str, default='configs/hdc_cnn_kfold.yaml',
                        help='Path to configuration file (default: configs/hdc_cnn_kfold.yaml)')
    parser.add_argument('--max-epochs', type=int, default=None,
                        help='Override number of epochs (default: use config value)')
    args = parser.parse_args()

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.set_float32_matmul_precision('high')

    print("HDC-CNN K-Fold Training")

    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = Config(str(config_path))
    if args.max_epochs is not None:
        config.cfg.training.epochs = args.max_epochs

    validate_config(config)

    data_loader = WLDataLoader(config.get('data.data_dir'), config.get('data.use_public_dataset'),
                               config.get('data.max_cosmologies', None))
    print(data_loader)
    labels_flat = data_loader.labels[:, :, :2].reshape(-1, 2)

    # Load or compute global normalization
    global_norm_path = Path(config.get('paths.output_dir')) / 'global_normalization.json'
    current_seed = config.get('data.noise_seed')

    if global_norm_path.exists():
        with open(global_norm_path) as f:
            global_norm_stats = json.load(f)

        need_recompute = (global_norm_stats.get('noise_seed') != current_seed or
                          global_norm_stats.get('noise_batch_size', NOISE_BATCH_SIZE) != NOISE_BATCH_SIZE)

        if need_recompute:
            kappa_noisy, global_norm_stats = compute_global_normalization(
                data_loader, current_seed, global_norm_path)
        else:
            kappa_flat = data_loader.kappa.reshape(-1, data_loader.img_height, data_loader.img_width)
            kappa_noisy = np.zeros_like(kappa_flat)
            for i in range(0, kappa_flat.shape[0], NOISE_BATCH_SIZE):
                end = min(i + NOISE_BATCH_SIZE, kappa_flat.shape[0])
                kappa_noisy[i:end] = Preprocessor.add_noise(
                    kappa_flat[i:end], data_loader.mask, data_loader.pixel_size_arcmin,
                    data_loader.galaxy_density_per_arcmin2, data_loader.shape_noise, current_seed + i)
    else:
        kappa_noisy, global_norm_stats = compute_global_normalization(
            data_loader, current_seed, global_norm_path)

    n_folds = config.get('data.n_folds')
    n_repeats = config.get('data.n_repeats')
    print(f"K-fold: {n_folds} folds x {n_repeats} repeats = {n_folds * n_repeats} models")

    # Load test data
    test_file = Path(config.get('data.data_dir')) / "WIDE12H_bin2_2arcmin_kappa_noisy_test.npy"
    mask_file = Path(config.get('data.data_dir')) / "WIDE12H_bin2_2arcmin_mask.npy"
    test_kappa_noisy = np.load(test_file).astype(np.float32)
    mask = np.load(mask_file).astype(np.float32)
    print(f"Test: {test_kappa_noisy.shape[0]} samples")

    if len(test_kappa_noisy.shape) == 2 and test_kappa_noisy.shape[-1] == np.count_nonzero(mask):
        test_kappa_full = np.zeros((test_kappa_noisy.shape[0], data_loader.img_height, data_loader.img_width), dtype=np.float32)
        test_kappa_full[:, mask.astype(bool)] = test_kappa_noisy
        test_kappa_noisy = test_kappa_full
    elif len(test_kappa_noisy.shape) == 3:
        expected = (test_kappa_noisy.shape[0], data_loader.img_height, data_loader.img_width)
        if test_kappa_noisy.shape != expected:
            raise ValueError(f"Test shape mismatch: {test_kappa_noisy.shape} vs {expected}")
    else:
        raise ValueError(f"Unexpected test shape: {test_kappa_noisy.shape}")

    preprocessor_test = Preprocessor()
    preprocessor_test.image_mean = global_norm_stats['image_mean']
    preprocessor_test.image_std = global_norm_stats['image_std']
    test_kappa_normalized = preprocessor_test.transform_images(test_kappa_noisy)

    from data.augmentations import get_val_transforms
    test_dataset = CosmologyDataset(test_kappa_normalized, None, get_val_transforms(),
                                    np.arange(len(test_kappa_normalized)))
    test_loader = DataLoader(test_dataset, batch_size=config.get('training.batch_size'),
                             shuffle=False, num_workers=8, pin_memory=True,
                             persistent_workers=True, prefetch_factor=2)

    all_results = []
    for repeat_id in range(n_repeats):
        repeat_seed = config.get('data.split_seed') + repeat_id * 1000
        kfold_splits = create_kfold_splits_by_systematics(
            data_loader.Ncosmo, data_loader.Nsys, n_folds, repeat_seed)

        for fold_id, (train_idx, val_idx) in enumerate(kfold_splits):
            result = train_single_fold(fold_id, repeat_id, train_idx, val_idx, kappa_noisy,
                                       labels_flat, global_norm_stats, config, test_loader)
            all_results.append(result)

    del kappa_noisy, labels_flat
    import gc
    gc.collect()

    scores = [r['best_score'] for r in all_results]
    summary = {
        'n_models': len(all_results), 'n_folds': n_folds, 'n_repeats': n_repeats,
        'results': all_results,
        'best_scores': {'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
                        'min': float(np.min(scores)), 'max': float(np.max(scores))}
    }

    summary_path = Path(config.get('paths.output_dir')) / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nScores: {summary['best_scores']['mean']:.4f} ± {summary['best_scores']['std']:.4f}")


def validate_config(config):
    required = [
        'data.data_dir', 'data.use_public_dataset', 'data.n_folds',
        'data.n_repeats', 'data.noise_seed', 'reproducibility.seed',
        'reproducibility.deterministic', 'reproducibility.benchmark',
        'training.batch_size', 'training.epochs', 'training.loss',
        'training.loss_params', 'training.optimizer', 'training.scheduler',
        'training.use_ema', 'training.beta_schedule', 'model.name', 'device'
    ]

    conditional = {
        'training.use_ema': ['training.ema_decay'],
        'training.lambda_train_schedule.enabled': [
            'training.lambda_train_schedule.start_epoch',
            'training.lambda_train_schedule.rampup_epochs',
            'training.lambda_train_schedule.target_lambda'
        ],
        'training.warmstart.enabled': ['training.warmstart.epochs'],
        'training.beta_schedule.enabled': [
            'training.beta_schedule.warmup_epochs',
            'training.beta_schedule.rampup_epochs',
            'training.beta_schedule.target_beta'
        ]
    }

    missing = [k for k in required if config.get(k) is None]
    for cond, keys in conditional.items():
        if config.get(cond):
            missing.extend(k for k in keys if config.get(k) is None)

    if missing:
        raise ValueError("Missing config keys: " + ", ".join(missing))

    print(" Config validated")


if __name__ == '__main__':
    main()
