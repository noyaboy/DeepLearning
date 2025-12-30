# Create final calibrated submission with MoG ensemble and optional isotonic calibration.

import sys
import argparse
from pathlib import Path
import numpy as np
import zipfile
import json
import pickle
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from utils.isotonic import IsotonicRegressionNumpy
from utils.io import read_predictions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create final calibrated submission'
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

    print(f"\nFinal submission: {args.run_name}")

    output_dir = Path(f'outputs/{args.run_name}')
    test_pred_dir = output_dir / 'test_predictions'
    calibration_dir = output_dir / 'calibration'
    submission_dir = Path('submissions')
    submission_dir.mkdir(parents=True, exist_ok=True)

    calibration_report_file = calibration_dir / 'calibration_report.json'
    with open(calibration_report_file) as f:
        calib_report = json.load(f)

    use_calibration = calib_report['decision']['use_calibration']
    print(f"Calibration: {'enabled' if use_calibration else 'disabled'}")

    if use_calibration:
        calibrators_file = calibration_dir / 'isotonic_regressors.pkl'
        with open(calibrators_file, 'rb') as f:
            calibrators = pickle.load(f)
        iso_Om = calibrators['iso_Om']
        iso_S8 = calibrators['iso_S8']

    test_files = sorted(test_pred_dir.glob('*.parquet'))
    print(f"Loading {len(test_files)} test predictions...")

    test_preds = []
    val_scores = []
    for f in test_files:
        df = read_predictions(f)
        test_preds.append(df)
        val_scores.append(df['val_score'].iloc[0])

    n_models = len(test_preds)
    weights = np.ones(n_models) / n_models

    mus = np.stack([df[['mu_Om', 'mu_S8']].values for df in test_preds], axis=0)
    vars = np.stack([df[['var_Om', 'var_S8']].values for df in test_preds], axis=0)

    ensemble_mu = np.sum(weights[:, None, None] * mus, axis=0)
    ensemble_var = np.sum(weights[:, None, None] * vars, axis=0)

    print(f"MoG ensemble: Om mean={ensemble_mu[:, 0].mean():.4f}, S8 mean={ensemble_mu[:, 1].mean():.4f}")

    if use_calibration:
        calibrated_var_Om = iso_Om.transform(ensemble_var[:, 0])
        calibrated_var_S8 = iso_S8.transform(ensemble_var[:, 1])
        calibrated_var_Om = np.maximum(calibrated_var_Om, 1e-10)
        calibrated_var_S8 = np.maximum(calibrated_var_S8, 1e-10)
        final_var = np.column_stack([calibrated_var_Om, calibrated_var_S8])
        print(f"Applied calibration")
    else:
        final_var = ensemble_var

    final_std = np.sqrt(final_var)

    submission = {
        'means': ensemble_mu.tolist(),
        'errorbars': final_std.tolist()
    }

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    calibration_suffix = "cal" if use_calibration else "nocal"
    submission_name = f"submission_kfold_mog_{calibration_suffix}_{timestamp}"

    result_json_path = submission_dir / f"{submission_name}.json"
    with open(result_json_path, 'w') as f:
        json.dump(submission, f)

    zip_path = submission_dir / f"{submission_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(result_json_path, 'result.json')

    result_json_path.unlink()

    print(f"Submission: {zip_path}")

    if use_calibration:
        calib_section = f"""OOF: uncal={calib_report['oof_scores']['uncalibrated']:.4f}, cal={calib_report['oof_scores']['calibrated']:.4f} ({calib_report['oof_scores']['improvement']:+.4f})"""
        expected_cv = calib_report['oof_scores']['calibrated']
    else:
        calib_section = "OOF: calibration skipped (improvement <= 0.01)"
        expected_cv = np.mean(val_scores)

    report_md = f"""Submission Report: {args.run_name}
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Config: 8 models (4 folds x 2 repeats), MoG ensemble
Calibration: {'isotonic' if use_calibration else 'none'}

{calib_section}

Val scores: mean={np.mean(val_scores):.4f}, std={np.std(val_scores):.4f}

Test predictions:
  Om: [{ensemble_mu[:, 0].min():.4f}, {ensemble_mu[:, 0].max():.4f}], mean={ensemble_mu[:, 0].mean():.4f}
  S8: [{ensemble_mu[:, 1].min():.4f}, {ensemble_mu[:, 1].max():.4f}], mean={ensemble_mu[:, 1].mean():.4f}
  Uncertainty: Om std={final_std[:, 0].mean():.4f}, S8 std={final_std[:, 1].mean():.4f}

Output: {zip_path.name} (4000 samples)
Expected CV: {expected_cv:.2f}
"""

    report_file = submission_dir / f"{submission_name}_report.md"
    with open(report_file, 'w') as f:
        f.write(report_md)

    print(f"Report: {report_file}")


if __name__ == '__main__':
    main()
