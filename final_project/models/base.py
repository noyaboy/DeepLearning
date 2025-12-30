# Base class for regression models with uncertainty.

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch):
        pass

    def predict(self, batch):
        self.eval()
        with torch.no_grad():
            out = self.forward(batch)
            return out['mean'], torch.exp(0.5 * out['log_var'])

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
