# Dataset wrapper for kappa maps.

import torch
import numpy as np
from torch.utils.data import Dataset


class CosmologyDataset(Dataset):
    def __init__(self, images, labels=None, image_transform=None, label_transform=None,
                 sample_ids=None):
        if isinstance(images, np.ndarray):
            self.images = torch.from_numpy(images).float()
        else:
            self.images = images

        if labels is not None:
            if isinstance(labels, np.ndarray):
                self.labels = torch.from_numpy(labels).float()
            else:
                self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None

        self.image_transform = image_transform
        self.label_transform = label_transform
        self.sample_ids = np.asarray(sample_ids) if sample_ids is not None else np.arange(len(images))
        self.is_test = labels is None

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.image_transform:
            img = self.image_transform(img)
        if img.dim() == 2:
            img = img.unsqueeze(0)

        out = {'image': img, 'sample_id': torch.tensor(self.sample_ids[idx], dtype=torch.long)}
        if not self.is_test:
            lbl = self.labels[idx]
            if self.label_transform:
                lbl = self.label_transform(lbl)
            out['label'] = lbl
        return out

    def __repr__(self):
        return f"CosmologyDataset(n={len(self)}, test={self.is_test})"
