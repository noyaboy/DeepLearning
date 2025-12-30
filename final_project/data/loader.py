# Loads kappa maps and labels for weak lensing analysis.

from pathlib import Path
import numpy as np


class DataLoader:
    def __init__(self, data_dir, use_public_dataset=True, img_height=1424, img_width=176,
                 max_cosmologies=None):
        self.data_dir = Path(data_dir)
        self.use_public_dataset = use_public_dataset
        self.img_height = img_height
        self.img_width = img_width
        self.max_cosmologies = max_cosmologies

        self.kappa, self.mask, self.labels = self._load_data()

        # survey parameters
        self.pixel_size_arcmin = 2.0
        self.galaxy_density_per_arcmin2 = 30.0
        self.shape_noise = 0.4

        self.Ncosmo = self.kappa.shape[0]
        self.Nsys = self.kappa.shape[1]
        self.ng = self.galaxy_density_per_arcmin2
        self.pixelsize_arcmin = self.pixel_size_arcmin

    def _load_data(self):
        if self.use_public_dataset:
            kappa_path = self.data_dir / "WIDE12H_bin2_2arcmin_kappa.npy"
            mask_path = self.data_dir / "WIDE12H_bin2_2arcmin_mask.npy"
            labels_path = self.data_dir / "label.npy"
        else:
            kappa_path = self.data_dir / "sampled_WIDE12H_bin2_2arcmin_kappa.npy"
            mask_path = self.data_dir / "WIDE12H_bin2_2arcmin_mask.npy"
            labels_path = self.data_dir / "sampled_label.npy"

        for p in [kappa_path, mask_path, labels_path]:
            if not p.exists():
                raise FileNotFoundError(f"Missing: {p}")

        print(f"    Loading {mask_path.name}...")
        mask = np.load(mask_path).astype(np.float32)
        print(f"    Mask shape: {mask.shape}")

        print(f"    Loading {kappa_path.name} (slow, ~30-60s)...")
        kappa_masked = np.load(kappa_path).astype(np.float32)
        print(f"    Kappa shape: {kappa_masked.shape}")

        # unpack masked format if needed
        if len(kappa_masked.shape) == 3 and kappa_masked.shape[-1] == np.count_nonzero(mask):
            print(f"    Unpacking to full images...")
            Ncosmo, Nsys, _ = kappa_masked.shape
            kappa = np.zeros((Ncosmo, Nsys, self.img_height, self.img_width), dtype=np.float32)
            kappa[:, :, mask.astype(bool)] = kappa_masked
            print(f"    Unpacked: {kappa.shape}")
        else:
            kappa = kappa_masked

        print(f"    Loading {labels_path.name}...")
        labels = np.load(labels_path).astype(np.float32)
        print(f"    Labels shape: {labels.shape}")

        assert kappa.shape[2:] == (self.img_height, self.img_width), "Image shape mismatch"
        assert mask.shape == (self.img_height, self.img_width), "Mask shape mismatch"

        if self.max_cosmologies is not None:
            kappa = kappa[:self.max_cosmologies]
            labels = labels[:self.max_cosmologies]
            print(f"    Using subset: {self.max_cosmologies} cosmologies")

        return kappa, mask, labels

    def load_test_data(self, test_dir):
        test_dir = Path(test_dir)
        kappa = np.load(test_dir / "kappa.npy").astype(np.float32)
        mask = np.load(test_dir / "mask.npy").astype(np.float32)
        return kappa, mask

    def __repr__(self):
        return f"DataLoader(ncosmo={self.Ncosmo}, nsys={self.Nsys}, shape={self.img_height}x{self.img_width})"
