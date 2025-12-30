#!/usr/bin/env python3
# Multi-GPU K-Fold Training Launcher. Runs multiple folds in parallel across GPUs.

import argparse
import subprocess
import os
from pathlib import Path
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Multi-GPU K-Fold Cross-Validation Training',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        required=True,
        help='Comma-separated GPU IDs (e.g., "0,1,2,3")'
    )
    parser.add_argument(
        '--n-folds',
        type=int,
        default=5,
        help='Number of folds (default: 5)'
    )
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=2,
        help='Number of repeats (default: 2)'
    )
    parser.add_argument(
        '--start-fold',
        type=int,
        default=0,
        help='Starting fold+repeat index (default: 0, range: 0-9 for 5 folds x 2 repeats)'
    )
    parser.add_argument(
        '--end-fold',
        type=int,
        default=None,
        help='Ending fold+repeat index (default: all remaining folds)'
    )

    return parser.parse_args()


def get_fold_combinations(n_folds, n_repeats, start_idx=0, end_idx=None):
    combinations = []
    for repeat_id in range(n_repeats):
        for fold_id in range(n_folds):
            combinations.append((repeat_id, fold_id))

    total = len(combinations)
    if end_idx is None:
        end_idx = total

    return combinations[start_idx:end_idx]


def run_single_fold(config_path, fold_id, repeat_id, gpu_id, logfile):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        sys.executable,
        '-u',
        'experiments/train_single_fold_worker.py',
        '--config', config_path,
        '--fold-id', str(fold_id),
        '--repeat-id', str(repeat_id)
    ]

    print(f"  [GPU {gpu_id}] Launching fold {fold_id}, repeat {repeat_id}")
    print(f"           Logging to: {logfile}")

    with open(logfile, 'w') as f:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent
        )

    return process


def main():
    args = parse_args()

    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    n_gpus = len(gpu_ids)

    try:
        import torch
        available_gpus = torch.cuda.device_count()
        for gpu_id in gpu_ids:
            if gpu_id >= available_gpus:
                print(f"ERROR: GPU {gpu_id} not available (only {available_gpus} GPUs detected)")
                print(f"Available GPU IDs: {list(range(available_gpus))}")
                sys.exit(1)
    except Exception as e:
        print(f"WARNING: Could not validate GPUs: {e}")

    print(f"\nMulti-GPU K-Fold Training")
    print(f"Config: {args.config}")
    print(f"GPUs: {gpu_ids}, Folds: {args.n_folds}, Repeats: {args.n_repeats}")

    print("\nComputing global normalization...")
    from experiments.train_kfold import Config, WLDataLoader, compute_global_normalization
    config = Config(args.config)
    global_norm_path = Path(config.get('paths.output_dir')) / 'global_normalization.json'

    if not global_norm_path.exists():
        print(f"  Global normalization not found, computing now...")
        data_loader = WLDataLoader(
            data_dir=config.get('data.data_dir'),
            use_public_dataset=config.get('data.use_public_dataset'),
            max_cosmologies=config.get('data.max_cosmologies', None)
        )
        kappa_noisy, global_norm_stats = compute_global_normalization(
            data_loader,
            noise_seed=config.get('data.noise_seed'),
            output_path=global_norm_path
        )
        print(f"  Saved: {global_norm_path}")
        del data_loader, kappa_noisy
    else:
        print(f"  Using existing: {global_norm_path}")
    print()

    log_dir = Path('logs/multigpu')
    log_dir.mkdir(parents=True, exist_ok=True)

    fold_combinations = get_fold_combinations(
        args.n_folds,
        args.n_repeats,
        args.start_fold,
        args.end_fold
    )

    print(f"\nTraining {len(fold_combinations)} folds across {n_gpus} GPUs...")
    print(f"Folds per GPU: ~{len(fold_combinations) / n_gpus:.1f}\n")

    running_processes = {}
    fold_queue = list(fold_combinations)
    completed = []
    failed = []

    try:
        workers_loading = set()
        workers_training = set()

        while fold_queue or running_processes:
            for gpu_id in gpu_ids:
                if gpu_id not in running_processes and fold_queue:
                    if len(workers_loading) > 0:
                        continue

                    repeat_id, fold_id = fold_queue.pop(0)
                    logfile = log_dir / f'fold_r{repeat_id}f{fold_id}_gpu{gpu_id}.log'

                    process = run_single_fold(
                        args.config,
                        fold_id,
                        repeat_id,
                        gpu_id,
                        logfile
                    )

                    running_processes[gpu_id] = (process, repeat_id, fold_id, logfile)
                    workers_loading.add(gpu_id)
                    print(f"  [GPU {gpu_id}] Worker launched, waiting for data loading...")

            for gpu_id in list(workers_loading):
                if gpu_id in running_processes:
                    _, repeat_id, fold_id, logfile = running_processes[gpu_id]

                    if logfile.exists():
                        try:
                            with open(logfile, 'r') as f:
                                log_content = f.read()
                                if '[6/6] Starting training...' in log_content:
                                    workers_loading.remove(gpu_id)
                                    workers_training.add(gpu_id)
                                    print(f"  GPU {gpu_id}: data loaded, starting next worker")
                        except Exception:
                            pass

            for gpu_id in list(running_processes.keys()):
                process, repeat_id, fold_id, logfile = running_processes[gpu_id]

                retcode = process.poll()
                if retcode is not None:
                    if retcode == 0:
                        print(f"  GPU {gpu_id}: completed fold {fold_id}, repeat {repeat_id}")
                        completed.append((repeat_id, fold_id))
                    else:
                        print(f"  GPU {gpu_id}: FAILED fold {fold_id}, repeat {repeat_id} (exit code: {retcode})")
                        print(f"           Check log: {logfile}")
                        failed.append((repeat_id, fold_id))

                    workers_loading.discard(gpu_id)
                    workers_training.discard(gpu_id)
                    del running_processes[gpu_id]

            if running_processes:
                print(f"\r  Active: {len(running_processes)}/{n_gpus} GPUs "
                      f"(Loading: {len(workers_loading)}, Training: {len(workers_training)}) | "
                      f"Completed: {len(completed)} | "
                      f"Remaining: {len(fold_queue)} | "
                      f"Failed: {len(failed)}", end='', flush=True)

            time.sleep(2)

        print("\n")  # New line after progress updates

    except KeyboardInterrupt:
        print("\n\nInterrupted by user!")
        print("Terminating running processes...")
        for gpu_id, (process, repeat_id, fold_id, _) in running_processes.items():
            print(f"  Killing fold {fold_id}, repeat {repeat_id} on GPU {gpu_id}")
            process.terminate()
            process.wait()
        sys.exit(1)

    print(f"\nTraining complete: {len(completed)}/{len(fold_combinations)} folds")

    if failed:
        print(f"Warning: {len(failed)} failed folds:")
        for repeat_id, fold_id in failed:
            print(f"   - Repeat {repeat_id}, Fold {fold_id}")
        print("Check log files in: logs/multigpu/")
        sys.exit(1)
    else:
        print("All folds completed")
        sys.exit(0)


if __name__ == '__main__':
    main()
