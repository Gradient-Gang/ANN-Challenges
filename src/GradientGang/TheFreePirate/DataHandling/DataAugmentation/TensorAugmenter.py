import torch
from abc import ABC, abstractmethod


class TensorAugmenter(ABC):
    @abstractmethod
    def augment(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        pass


class SequentialAugmenter(TensorAugmenter):
    def __init__(
        self, nCopies: int, keepOriginal: bool, augmenters: list["TensorAugmenter"]
    ):
        if nCopies < 0:
            raise ValueError("nCopies must be >= 0")
        self.nCopies = nCopies

        self.keepOriginal = keepOriginal

        if augmenters is None or len(augmenters) == 0:
            raise ValueError("At least one augmenter must be provided")
        self.augmenters = augmenters

    def augment(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        augmentedTensor = tensor.clone() if self.keepOriginal else torch.empty(0)

        for _ in range(self.nCopies):
            augmentedSample = tensor
            for augmenter in self.augmenters:
                augmentedSample = augmenter.augment(augmentedSample, dim=dim)
            augmentedTensor = (
                torch.cat((augmentedTensor, augmentedSample), dim=0)
                if augmentedTensor.numel() > 0
                else augmentedSample
            )

        return augmentedTensor


class IdentityAugmenter(TensorAugmenter):
    def augment(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return tensor.clone()


class JitterAugmenter(TensorAugmenter):
    def __init__(self, jitterStdDev: float):
        self.jitterStdDev = jitterStdDev

    def augment(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        noise = torch.randn_like(tensor) * self.jitterStdDev
        return tensor + noise


class ScaleAugmenter(TensorAugmenter):
    def __init__(self, rangeScale: float):
        self.scaleMin = 1.0 - rangeScale
        self.scaleMax = 1.0 + rangeScale

    def augment(self, tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        scaleFactor = (
            torch.rand(1).item() * (self.scaleMax - self.scaleMin) + self.scaleMin
        )
        return tensor * scaleFactor


class OffsetAugmenter(TensorAugmenter):
    def __init__(self, rangeOffset: float):
        self.offsetMin = -rangeOffset
        self.offsetMax = rangeOffset

    def augment(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        # build an offset with the same shape as `tensor` except at `dim`
        sample = tensor.select(dim, 0)  # shape without the time dim
        offsetValue = (
            torch.rand_like(sample) * (self.offsetMax - self.offsetMin) + self.offsetMin
        )
        # reinstate the time dim so broadcasting matches the original tensor
        offsetValue = offsetValue.unsqueeze(dim)
        return tensor + offsetValue


class TimeWarpAugmenter(TensorAugmenter):
    def __init__(self, maxWarpFraction: float):
        self.maxWarpFraction = maxWarpFraction

    def augment(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        numTimeSteps = tensor.size(dim)
        warpFraction = (torch.rand(1).item() * 2 - 1) * self.maxWarpFraction
        warpAmount = int(numTimeSteps * warpFraction)

        if warpAmount == 0:
            return tensor

        # Create new time indices with warping
        originalIndices = torch.arange(numTimeSteps)
        warpedIndices = (
            originalIndices + torch.linspace(0, warpAmount, numTimeSteps).long()
        )

        # Clamp indices to valid range
        warpedIndices = torch.clamp(warpedIndices, 0, numTimeSteps - 1)

        # Apply warping
        warpedTensor = tensor.index_select(dim, warpedIndices)

        return warpedTensor


class WindowingAugmenter(TensorAugmenter):
    def __init__(self, windowSize: int, stride: int):
        self.windowSize = windowSize
        self.stride = stride

    def augment(self, tensor: torch.Tensor, dim: int = 1) -> torch.Tensor:
        # output (..., num_windows, window_size, ...)

        numTimeSteps = tensor.size(dim)
        windows = []

        for start in range(0, numTimeSteps - self.windowSize + 1, self.stride):
            end = start + self.windowSize
            window = tensor.narrow(dim, start, self.windowSize)

            # Insert new dimension BEFORE `dim`
            window = window.unsqueeze(dim)

            windows.append(window)

        # concatenate along the new dim
        return torch.cat(windows, dim=dim)
