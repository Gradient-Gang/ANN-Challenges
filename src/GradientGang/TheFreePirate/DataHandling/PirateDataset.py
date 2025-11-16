import pytorch_lightning as L
import torch
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset,
    random_split,
    Subset,
)
import pandas as pd
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from sklearn.model_selection import StratifiedKFold
from .DataAugmentation.TensorAugmenter import TensorAugmenter


class PirateDataset(Dataset):
    def __init__(
        self,
        timeSeriesTensor: torch.Tensor,  # [numSamples, numWindows, numFeatures, timesteps]
        globalFeaturesTensor: torch.Tensor,  # [numSamples, numGlobalFeatures]
        labelsTensor: torch.Tensor,  # [numSamples]
    ):
        if timeSeriesTensor.ndim != 4:
            raise ValueError(
                f"timeSeriesTensor must be 4D (numSamples, numWindows, numFeatures, timesteps), got {timeSeriesTensor.ndim}D"
            )

        if globalFeaturesTensor.ndim != 2:
            raise ValueError(
                f"globalFeaturesTensor must be 2D (numSamples, numGlobalFeatures), got {globalFeaturesTensor.ndim}D"
            )

        if labelsTensor.ndim != 1:
            raise ValueError(
                f"labelsTensor must be 1D (numSamples), got {labelsTensor.ndim}D"
            )

        if (
            timeSeriesTensor.shape[0] != globalFeaturesTensor.shape[0]
            or timeSeriesTensor.shape[0] != labelsTensor.shape[0]
        ):
            raise ValueError(
                "Mismatched number of samples between timeSeriesTensor, globalFeaturesTensor, and labelsTensor"
            )

        self.timeSeriesTensor = timeSeriesTensor
        self.globalFeaturesTensor = globalFeaturesTensor
        self.labelsTensor = labelsTensor
        self.numSamples = timeSeriesTensor.shape[0]

    def __len__(self):
        return self.numSamples

    def __getitem__(self, idx):
        return (
            self.timeSeriesTensor[idx],  # [numIdx, numWindows, numFeatures, timesteps]
            self.globalFeaturesTensor[idx],  # [numIdx, numGlobalFeatures]
            self.labelsTensor[idx],  # [numIdx]
        )
