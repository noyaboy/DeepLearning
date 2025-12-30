# Create performance-weighted ensemble as alternative to stacker.

import sys
import argparse
from pathlib import Path
import numpy as np
import json

sys.path.insert(0, str(Path(__file__).parent))
from utils.io import read_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create performance-weighted ensemble'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Run name (e.g., hdc_cnn_kfold_a100_stable)'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='softmax',
        choices=['uniform', 'score', 'softmax'],
        help='Weighting method: uniform, score (linear), or softmax'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=2.0,
        help='Temperature for softmax weighting (higher = more uniform)'
    )
    return parser.parse_args()


def compute_weights_uniform(scores):
    return np.ones(len(scores)) / len(scores)


def compute_weights_score_linear(scores):
    scores = np.array(scores)
    weights = scores / scores.sum()
    return weights


def compute_weights_softmax(scores, temperature=2.0):
    scores = np.array(scores)
    scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    exp_scores = np.exp(scores_norm / temperature)
    weights = exp_scores / exp_scores.sum()
    return weights


def evaluate_weighted_ensemble(test_preds, weights):
    n_models = len(test_preds)
    n_samples = len(test_preds[0])

    mus = np.stack([df[['mu_Om', 'mu_S8']].values for df in test_preds], axis=0)
    vars = np.stack([df[['var_Om', 'var_S8']].values for df in test_preds], axis=0)

    ensemble_mu = np.sum(weights[:, None, None] * mus, axis=0)
    ensemble_var = np.sum(weights[:, None, None] * vars, axis=0)

    return ensemble_mu, ensemble_var


def main():
    args = parse_args()

    print(f"\nPerformance-weighted ensemble")
    print(f"Run: {args.run_name}, method: {args.method}" +
          (f", T={args.temperature}" if args.method == 'softmax' else ""))

    output_dir = Path(f'outputs/{args.run_name}')
    test_pred_dir = output_dir / 'test_predictions'
    stacker_dir = output_dir / 'stacker'
    stacker_dir.mkdir(parents=True, exist_ok=True)

    test_files = sorted(test_pred_dir.glob('*.parquet'))
    print(f"\nLoading {len(test_files)} test predictions...")

    test_preds = []
    val_scores = []
    model_names = []

    for f in test_files:
        df = read_predictions(f)
        test_preds.append(df)
        val_scores.append(df['val_score'].iloc[0])
        model_names.append(f.stem.replace('test_', ''))

    val_scores = np.array(val_scores)
    print(f"Val scores: mean={val_scores.mean():.4f}, std={val_scores.std():.4f}")

    print("\nComputing weights...")

    weights_uniform = compute_weights_uniform(val_scores)
    weights_score = compute_weights_score_linear(val_scores)
    weights_softmax = compute_weights_softmax(val_scores, args.temperature)

    mog_mu, mog_var = evaluate_weighted_ensemble(test_preds, weights_uniform)
    score_mu, score_var = evaluate_weighted_ensemble(test_preds, weights_score)
    softmax_mu, softmax_var = evaluate_weighted_ensemble(test_preds, weights_softmax)

    print(f"Uniform: Om mean={mog_mu[:, 0].mean():.4f}, S8 mean={mog_mu[:, 1].mean():.4f}")
    print(f"Score:   Om mean={score_mu[:, 0].mean():.4f}, S8 mean={score_mu[:, 1].mean():.4f}")
    print(f"Softmax: Om mean={softmax_mu[:, 0].mean():.4f}, S8 mean={softmax_mu[:, 1].mean():.4f}")

    ref_path = output_dir / 'reference_submission' / 'reference_predictions.parquet'
    ref_path_pkl = output_dir / 'reference_submission' / 'reference_predictions.pkl'
    if ref_path.exists() or ref_path_pkl.exists():
        ref_df = read_predictions(ref_path if ref_path.exists() else ref_path_pkl)
        mae_softmax_Om = np.abs(softmax_mu[:, 0] - ref_df['mu_Om'].values).mean()
        mae_softmax_S8 = np.abs(softmax_mu[:, 1] - ref_df['mu_S8'].values).mean()
        print(f"MAE vs reference: Om={mae_softmax_Om:.4f}, S8={mae_softmax_S8:.4f}")

    score_cv = val_scores.std() / val_scores.mean()

    if score_cv > 0.05:
        selected_method = 'softmax'
        selected_weights = weights_softmax
        selected_mu = softmax_mu
        selected_var = softmax_var
        reasoning = f"CV={score_cv:.3f} > 0.05 -> softmax"
    else:
        selected_method = 'uniform'
        selected_weights = weights_uniform
        selected_mu = mog_mu
        selected_var = mog_var
        reasoning = f"CV={score_cv:.3f} <= 0.05 -> uniform"

    print(f"\nSelected: {selected_method} ({reasoning})")

    results = {
        'validation_scores': {
            'models': dict(zip(model_names, val_scores.tolist())),
            'mean': float(val_scores.mean()),
            'std': float(val_scores.std()),
            'cv': float(score_cv)
        },
        'weights': {
            'uniform': dict(zip(model_names, weights_uniform.tolist())),
            'score_linear': dict(zip(model_names, weights_score.tolist())),
            'softmax': dict(zip(model_names, weights_softmax.tolist())),
            'softmax_temperature': args.temperature
        },
        'selection': {
            'method': selected_method,
            'weights': dict(zip(model_names, selected_weights.tolist())),
            'reasoning': reasoning
        },
        'statistics': {
            'mog': {
                'Om_range': [float(mog_mu[:, 0].min()), float(mog_mu[:, 0].max())],
                'S8_range': [float(mog_mu[:, 1].min()), float(mog_mu[:, 1].max())],
                'Om_mean': float(mog_mu[:, 0].mean()),
                'S8_mean': float(mog_mu[:, 1].mean())
            },
            'selected': {
                'Om_range': [float(selected_mu[:, 0].min()), float(selected_mu[:, 0].max())],
                'S8_range': [float(selected_mu[:, 1].min()), float(selected_mu[:, 1].max())],
                'Om_mean': float(selected_mu[:, 0].mean()),
                'S8_mean': float(selected_mu[:, 1].mean())
            }
        }
    }

    results_file = stacker_dir / 'ensemble_weights.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Saved: {results_file}")


if __name__ == '__main__':
    main()
