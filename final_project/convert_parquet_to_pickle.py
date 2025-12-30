# Convert parquet files to pickle format for environments without pyarrow.

import argparse
from pathlib import Path
import pandas as pd
import pickle


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert parquet files to pickle format'
    )
    parser.add_argument(
        '--run-name',
        type=str,
        required=True,
        help='Run name (e.g., hdc_cnn_kfold_a100_stable)'
    )
    return parser.parse_args()


def convert_file(parquet_path):
    pkl_path = parquet_path.with_suffix('.pkl')

    try:
        df = pd.read_parquet(parquet_path)
        with open(pkl_path, 'wb') as f:
            pickle.dump(df, f)
        return pkl_path
    except Exception as e:
        print(f"  ERROR: Failed to convert {parquet_path.name}: {e}")
        return None


def main():
    args = parse_args()

    print(f"\nConvert parquet to pickle: {args.run_name}")

    output_dir = Path(f'outputs/{args.run_name}')

    if not output_dir.exists():
        print(f"\nERROR: Output directory not found: {output_dir}")
        return

    parquet_files = list(output_dir.rglob('*.parquet'))
    print(f"\nFound {len(parquet_files)} parquet files")

    if not parquet_files:
        print("No parquet files to convert.")
        return

    print("\nConverting files...")
    converted = 0
    failed = 0

    for pq_file in parquet_files:
        rel_path = pq_file.relative_to(output_dir)
        pkl_path = convert_file(pq_file)

        if pkl_path:
            print(f"  {rel_path} -> {pkl_path.name}")
            converted += 1
        else:
            failed += 1

    print(f"\nDone: {converted} converted, {failed} failed")

    if converted > 0:
        print("\nPickle files created. The inference scripts will now use these")
        print("as fallback when pyarrow is not available.")


if __name__ == '__main__':
    main()
