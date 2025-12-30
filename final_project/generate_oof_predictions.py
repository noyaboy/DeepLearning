# OOF predictions are generated during training by train_kfold.py.

import sys
from pathlib import Path

def main():
    print("OOF predictions are generated during training by train_kfold.py.")
    print("Expected output: outputs/hdc_cnn_kfold_a100_stable/oof/*.parquet (8 files)")
    print("\nOOF predictions should be generated during training.")
    print("Check: outputs/hdc_cnn_kfold_a100_stable/oof/")

    oof_dir = Path('outputs/hdc_cnn_kfold_a100_stable/oof')
    if oof_dir.exists():
        oof_files = list(oof_dir.glob('*.parquet'))
        print(f"\nFound {len(oof_files)} OOF files:")
        for f in sorted(oof_files):
            print(f"  - {f.name}")
    else:
        print(f"\nOOF directory not found: {oof_dir}")
        print("  Run training first")

if __name__ == '__main__':
    main()
