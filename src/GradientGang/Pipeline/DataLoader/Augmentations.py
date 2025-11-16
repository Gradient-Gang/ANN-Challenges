"""
Time Series Data Augmentation Module

Provides augmentation techniques for time series data following SOLID principles.
All augmentations are reproducible via seed parameter.

Performance: All operations use PyTorch for GPU acceleration and minimal overhead.
"""

import torch
from abc import ABC, abstractmethod


class BaseAugmentation(ABC):
    """
    Abstract base class for time series augmentations.
    
    Following SOLID principles:
    - Single Responsibility: Each augmentation handles one transformation
    - Open/Closed: Easy to extend with new augmentations
    - Liskov Substitution: All augmentations are interchangeable
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize augmentation with random seed for reproducibility.
        
        Args:
            seed (int): Random seed for reproducible augmentation
        """
        self.seed = seed
    
    @abstractmethod
    def apply(self, data: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """
        Apply augmentation to time series data.
        
        Args:
            data (torch.Tensor): Time series data of shape (channels, time_steps)
            sample_idx (int): Sample index for reproducible seeding
            
        Returns:
            torch.Tensor: Augmented time series data
        """
        pass
    
    def _get_generator(self, sample_idx: int) -> torch.Generator:
        """
        Create a seeded random number generator for this sample.
        
        Args:
            sample_idx (int): Sample index for unique seed
            
        Returns:
            torch.Generator: Seeded random number generator
        """
        # Combine base seed with sample index for unique but reproducible randomness
        sample_seed = self.seed + sample_idx
        generator = torch.Generator()
        generator.manual_seed(sample_seed)
        return generator


class JitterAugmentation(BaseAugmentation):
    """
    Add Gaussian noise to time series data.
    
    Good for: Robustness to sensor noise, handling measurement errors
    
    Formula: x' = x + ε, where ε ~ N(0, σ²)
    """
    
    def __init__(self, strength: float = 0.05, seed: int = 42):
        """
        Initialize jitter augmentation.
        
        Args:
            strength (float): Standard deviation of Gaussian noise (0.01-0.1 recommended)
            seed (int): Random seed for reproducibility
        """
        super().__init__(seed)
        self.strength = strength
    
    def apply(self, data: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """
        Apply jittering (additive Gaussian noise) to time series.
        
        Args:
            data (torch.Tensor): Shape (channels, time_steps)
            sample_idx (int): Sample index for seeding
            
        Returns:
            torch.Tensor: Jittered time series
        """
        generator = self._get_generator(sample_idx)
        
        # Generate noise with same shape as data (fast, on device)
        noise = torch.randn(data.shape, dtype=data.dtype, device=data.device, generator=generator) * self.strength
        
        return data + noise


class ScalingAugmentation(BaseAugmentation):
    """
    Scale time series by random factor.
    
    Good for: Handling amplitude variations, intensity differences
    
    Formula: x' = x * α, where α ~ U[1-range, 1+range]
    """
    
    def __init__(self, scaling_range: float = 0.1, seed: int = 42):
        """
        Initialize scaling augmentation.
        
        Args:
            scaling_range (float): Range for scaling factor (0.05-0.15 recommended)
                                  Factor will be sampled from [1-range, 1+range]
            seed (int): Random seed for reproducibility
        """
        super().__init__(seed)
        self.scaling_range = scaling_range
    
    def apply(self, data: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """
        Apply amplitude scaling to time series.
        
        Args:
            data (torch.Tensor): Shape (channels, time_steps)
            sample_idx (int): Sample index for seeding
            
        Returns:
            torch.Tensor: Scaled time series
        """
        generator = self._get_generator(sample_idx)
        
        # Sample random scaling factor (fast, on device)
        scale_factor = torch.rand(1, dtype=data.dtype, device=data.device, generator=generator).item()
        scale_factor = (1 - self.scaling_range) + scale_factor * (2 * self.scaling_range)
        
        return data * scale_factor


class TimeWarpingAugmentation(BaseAugmentation):
    """
    Warp time axis using smooth random distortion.
    
    Good for: Handling timing variations, different speeds of phenomena
    
    Implementation: Uses smooth random displacement field
    """
    
    def __init__(self, strength: float = 0.2, seed: int = 42):
        """
        Initialize time warping augmentation.
        
        Args:
            strength (float): Warping strength (0.1-0.5 recommended)
                            Controls magnitude of time distortion
            seed (int): Random seed for reproducibility
        """
        super().__init__(seed)
        self.strength = strength
        self._indices_cache = {}
        self._generator_cache = {}
    
    def apply(self, data: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """
        Apply time warping to time series using linear interpolation.
        
        OPTIMIZED: Uses cached generators and index tensors.
        
        Args:
            data (torch.Tensor): Shape (channels, time_steps)
            sample_idx (int): Sample index for seeding
            
        Returns:
            torch.Tensor: Time-warped time series
        """
        # OPTIMIZATION: Cache generator per device
        device = data.device
        if device not in self._generator_cache:
            gen = torch.Generator(device=device)
            gen.manual_seed(self.seed + sample_idx)
            self._generator_cache[device] = gen
        else:
            self._generator_cache[device].manual_seed(self.seed + sample_idx)
        
        channels, time_steps = data.shape
        
        # OPTIMIZATION: Cache original indices tensor to avoid recreating it
        cache_key = (device, time_steps)
        if cache_key not in self._indices_cache:
            self._indices_cache[cache_key] = torch.arange(time_steps, dtype=torch.float32, device=device)
        original_indices = self._indices_cache[cache_key]
        
        # Generate smooth random warp using cumulative sum (pure PyTorch)
        warp = torch.randn(time_steps, dtype=data.dtype, device=device, generator=self._generator_cache[device]) * self.strength
        warp = torch.cumsum(warp, dim=0)
        warp.sub_(warp.mean())  # In-place: Center around 0
        warp.div_(warp.std() + 1e-8).mul_(self.strength)  # In-place: Normalize
        
        # Create warped sampling positions
        sample_positions = original_indices + warp * time_steps
        
        # Clip to valid range (in-place for speed)
        sample_positions.clamp_(0, time_steps - 1)
        
        # Linear interpolation for ALL channels at once (fully vectorized, NO loops!)
        # Get integer and fractional parts
        indices_floor = sample_positions.long()
        indices_ceil = torch.clamp(indices_floor + 1, max=time_steps - 1)
        weights = sample_positions - indices_floor.float()
        
        # Vectorized interpolation: process all channels simultaneously
        # Shape: data is (channels, time_steps), indices are (time_steps,)
        # Use advanced indexing to gather values for all channels at once
        values_floor = data[:, indices_floor]  # (channels, time_steps)
        values_ceil = data[:, indices_ceil]    # (channels, time_steps)
        
        # Broadcasting: weights is (time_steps,) -> (1, time_steps) for broadcasting
        weights = weights.unsqueeze(0)  # (1, time_steps)
        
        # Interpolate all channels in one operation (pure PyTorch, GPU-accelerated)
        warped_data = values_floor * (1 - weights) + values_ceil * weights
        
        return warped_data


class AugmentationPipeline:
    """
    Pipeline to apply multiple augmentations sequentially.
    
    Follows Composite pattern for combining multiple augmentations.
    """
    
    def __init__(self, augmentations: list[BaseAugmentation]):
        """
        Initialize augmentation pipeline.
        
        Args:
            augmentations (list[BaseAugmentation]): List of augmentations to apply
        """
        self.augmentations = augmentations
    
    def apply(self, data: torch.Tensor, sample_idx: int) -> torch.Tensor:
        """
        Apply all augmentations in sequence.
        
        Args:
            data (torch.Tensor): Time series data of shape (channels, time_steps)
            sample_idx (int): Sample index for seeding
            
        Returns:
            torch.Tensor: Augmented time series data
        """
        augmented_data = data
        for augmentation in self.augmentations:
            augmented_data = augmentation.apply(augmented_data, sample_idx)
        return augmented_data
    
    @classmethod
    def from_config(cls, config: dict, seed: int = 42) -> "AugmentationPipeline":
        """
        Create augmentation pipeline from configuration dictionary.
        
        Args:
            config (dict): Augmentation configuration with keys:
                - jitter_enabled (bool): Enable jitter augmentation
                - jitter_strength (float): Jitter noise strength
                - scaling_enabled (bool): Enable scaling augmentation
                - scaling_range (float): Scaling factor range
                - time_warp_enabled (bool): Enable time warping
                - time_warp_strength (float): Time warping strength
            seed (int): Random seed for reproducibility
            
        Returns:
            AugmentationPipeline: Configured augmentation pipeline
        """
        augmentations = []
        
        # Add jitter if enabled
        if config.get("jitter_enabled", False):
            jitter_strength = config.get("jitter_strength", 0.05)
            augmentations.append(JitterAugmentation(strength=jitter_strength, seed=seed))
        
        # Add scaling if enabled
        if config.get("scaling_enabled", False):
            scaling_range = config.get("scaling_range", 0.1)
            augmentations.append(ScalingAugmentation(scaling_range=scaling_range, seed=seed))
        
        # Add time warping if enabled
        if config.get("time_warp_enabled", False):
            time_warp_strength = config.get("time_warp_strength", 0.2)
            augmentations.append(TimeWarpingAugmentation(strength=time_warp_strength, seed=seed))
        
        return cls(augmentations)
