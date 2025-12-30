# Apply isotonic calibration to improve uncertainty estimates.

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
import json

sys.path.insert(0, str(Path(__file__).parent))
from evaluation.scoring import CompetitionScorer
from utils.isotonic import IsotonicRegressionNumpy
from utils.io import read_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Apply isotonic calibration'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Run name (e.g., hdc_cnn_kfold_a100_stable)'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\nIsotonic calibration: {args.run_name}")

    output_dir = Path(f'outputs/{args.run_name}')
    oof_dir = output_dir / 'oof'
    calibration_dir = output_dir / 'calibration'
    calibration_dir.mkdir(parents=True, exist_ok=True)

    oof_files = sorted(oof_dir.glob('*.parquet'))
    print(f"Loading {len(oof_files)} OOF files...")

    oof_dfs = []
    for f in oof_files:
        df = read_predictions(f)
        oof_dfs.append(df)

    oof_all = pd.concat(oof_dfs, ignore_index=True)
    print(f"  Total: {len(oof_all)} predictions")

    y_true_Om = oof_all['y_true_Om'].values
    y_true_S8 = oof_all['y_true_S8'].values
    mu_Om = oof_all['mu_Om'].values
    mu_S8 = oof_all['mu_S8'].values
    var_Om = oof_all['var_Om'].values
    var_S8 = oof_all['var_S8'].values

    squared_residuals_Om = (y_true_Om - mu_Om) ** 2
    squared_residuals_S8 = (y_true_S8 - mu_S8) ** 2

    std_Om = np.sqrt(var_Om)
    std_S8 = np.sqrt(var_S8)
    score_uncalibrated = CompetitionScorer.score_phase1(
        np.column_stack([y_true_Om, y_true_S8]),
        np.column_stack([mu_Om, mu_S8]),
        np.column_stack([std_Om, std_S8])
    )
    print(f"Uncalibrated OOF score: {score_uncalibrated:.4f}")

    print("Fitting isotonic regressors...")

    iso_Om = IsotonicRegressionNumpy(out_of_bounds='clip')
    iso_S8 = IsotonicRegressionNumpy(out_of_bounds='clip')

    iso_Om.fit(var_Om, squared_residuals_Om)
    iso_S8.fit(var_S8, squared_residuals_S8)
    print(f"  Om: {len(iso_Om.X_thresholds_)} control points, S8: {len(iso_S8.X_thresholds_)} control points")

    calibrated_var_Om = iso_Om.transform(var_Om)
    calibrated_var_S8 = iso_S8.transform(var_S8)
    calibrated_var_Om = np.maximum(calibrated_var_Om, 1e-10)
    calibrated_var_S8 = np.maximum(calibrated_var_S8, 1e-10)

    calibrated_std_Om = np.sqrt(calibrated_var_Om)
    calibrated_std_S8 = np.sqrt(calibrated_var_S8)
    score_calibrated = CompetitionScorer.score_phase1(
        np.column_stack([y_true_Om, y_true_S8]),
        np.column_stack([mu_Om, mu_S8]),
        np.column_stack([calibrated_std_Om, calibrated_std_S8])
    )

    improvement = score_calibrated - score_uncalibrated

    print(f"Calibrated: {score_calibrated:.4f} (improvement: {improvement:+.4f})")

    use_calibration = improvement > 0.01
    decision = "use" if use_calibration else "skip"
    reasoning = f"improvement {'>' if use_calibration else '<='} 0.01"

    print(f"Decision: {decision} calibration ({reasoning})")

    calibrators_file = calibration_dir / 'isotonic_regressors.pkl'
    with open(calibrators_file, 'wb') as f:
        pickle.dump({'iso_Om': iso_Om, 'iso_S8': iso_S8}, f)
    print(f"Saved: {calibrators_file}")

    report = {
        'method': 'isotonic_regression',
        'oof_scores': {
            'uncalibrated': float(score_uncalibrated),
            'calibrated': float(score_calibrated),
            'improvement': float(improvement)
        },
        'decision': {
            'use_calibration': use_calibration,
            'reasoning': reasoning
        },
        'calibrator_info': {
            'Om': {
                'n_control_points': int(len(iso_Om.X_thresholds_)),
                'input_range': [float(var_Om.min()), float(var_Om.max())],
                'output_range': [float(calibrated_var_Om.min()), float(calibrated_var_Om.max())]
            },
            'S8': {
                'n_control_points': int(len(iso_S8.X_thresholds_)),
                'input_range': [float(var_S8.min()), float(var_S8.max())],
                'output_range': [float(calibrated_var_S8.min()), float(calibrated_var_S8.max())]
            }
        },
        'statistics': {
            'Om': {
                'mean_var_uncalibrated': float(var_Om.mean()),
                'mean_var_calibrated': float(calibrated_var_Om.mean()),
                'mean_squared_residual': float(squared_residuals_Om.mean())
            },
            'S8': {
                'mean_var_uncalibrated': float(var_S8.mean()),
                'mean_var_calibrated': float(calibrated_var_S8.mean()),
                'mean_squared_residual': float(squared_residuals_S8.mean())
            }
        }
    }

    report_file = calibration_dir / 'calibration_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_file}")


if __name__ == '__main__':
    main()
