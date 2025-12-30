# Noise injection and normalization for kappa maps.

import numpy as np
from sklearn.preprocessing import StandardScaler


class Preprocessor:
    @staticmethod
    def add_noise(kappa, mask, pixel_size_arcmin, galaxy_density, shape_noise, seed):
        np.random.seed(seed)
        noise_std = shape_noise / np.sqrt(2 * galaxy_density * pixel_size_arcmin ** 2)
        noise = np.random.normal(0, noise_std, kappa.shape).astype(np.float32)
        return (kappa + noise * mask).astype(np.float32)

    def __init__(self):
        self.image_mean = None
        self.image_std = None
        self.label_scaler = StandardScaler()

    def fit_transform_images(self, images, batch_size=5000):
        n = images.shape[0]
        npix = images[0].size

        # compute mean in batches
        self.image_mean = 0.0
        for i in range(0, n, batch_size):
            batch = images[i:min(i+batch_size, n)]
            self.image_mean += batch.sum()
        self.image_mean /= (n * npix)

        # compute std in batches
        self.image_std = 0.0
        for i in range(0, n, batch_size):
            batch = images[i:min(i+batch_size, n)]
            for j in range(0, len(batch), 1000):
                sub = batch[j:min(j+1000, len(batch))]
                self.image_std += np.sum((sub - self.image_mean) ** 2)
        self.image_std = np.sqrt(self.image_std / (n * npix))

        # normalize
        result = images.copy()
        for i in range(0, n, 1000):
            result[i:min(i+1000, n)] = (result[i:min(i+1000, n)] - self.image_mean) / (self.image_std + 1e-8)
        return result

    def transform_images(self, images, batch_size=5000, inplace=False):
        if self.image_mean is None:
            raise ValueError("Not fitted yet")
        result = images if inplace else images.copy()
        n = result.shape[0]
        for i in range(0, n, 1000):
            result[i:min(i+1000, n)] = (result[i:min(i+1000, n)] - self.image_mean) / (self.image_std + 1e-8)
        return result

    def fit_transform_labels(self, labels):
        self.label_scaler.fit(labels)
        return self.label_scaler.transform(labels)

    def transform_labels(self, labels):
        return self.label_scaler.transform(labels)

    def inverse_transform_labels(self, labels):
        return self.label_scaler.inverse_transform(labels)

    def get_label_scale(self):
        return self.label_scaler.scale_
