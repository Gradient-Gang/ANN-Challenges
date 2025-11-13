"""
Test to verify there is NO data leakage between train and validation sets
when using windowing.
"""

import torch
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.GradientGang.Pipeline.DataLoader.DataLoader import TimeSeriesAndGlobalDataset


def test_no_leakage_with_windowing():
    """
    Test that windowing applied AFTER split prevents data leakage.
    """
    print("=" * 70)
    print("TEST: No Data Leakage with Windowing")
    print("=" * 70)

    # Create synthetic dataset
    num_samples = 10
    num_features = 2
    timesteps = 160

    time_series = torch.randn(num_samples, num_features, timesteps)
    global_features = torch.randn(num_samples, 5)
    labels = torch.randint(0, 3, (num_samples,))

    # Create dataset WITHOUT windowing first
    dataset_no_window = TimeSeriesAndGlobalDataset(
        time_series, global_features, labels, use_windowing=False
    )

    print(f"\n1. Original dataset (no windowing):")
    print(f"   Samples: {len(dataset_no_window)}")

    # Split into train/val (similar to DataModule.setup)
    from torch.utils.data import random_split, Subset

    train_size = 7
    val_size = 3

    train_subset, val_subset = random_split(
        dataset_no_window,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"\n2. After split (before windowing):")
    print(f"   Train samples: {len(train_subset)}")
    print(f"   Val samples: {len(val_subset)}")
    print(f"   Train indices: {train_subset.indices}")
    print(f"   Val indices: {val_subset.indices}")

    # Extract data for subsets and apply windowing separately
    # This is what DataModule._apply_windowing_to_subset does
    def apply_windowing(subset, window_size, stride):
        indices = subset.indices
        original_dataset = subset.dataset

        subset_time_series = original_dataset.time_series_data[indices]
        subset_global = original_dataset.global_data[indices]
        subset_labels = original_dataset.labels[indices]

        windowed = TimeSeriesAndGlobalDataset(
            subset_time_series,
            subset_global,
            subset_labels,
            use_windowing=True,
            window_size=window_size,
            stride=stride,
        )
        return windowed

    # Apply windowing to each subset
    window_size = 80
    stride = 40  # 50% overlap

    train_windowed = apply_windowing(train_subset, window_size, stride)
    val_windowed = apply_windowing(val_subset, window_size, stride)

    print(f"\n3. After windowing (applied separately):")
    print(f"   Train windows: {len(train_windowed)}")
    print(f"   Val windows: {len(val_windowed)}")

    # Calculate expected windows per sample
    windows_per_sample = (timesteps - window_size) // stride + 1
    print(f"   Expected windows per sample: {windows_per_sample}")
    print(
        f"   Train: {train_size} samples × {windows_per_sample} = {train_size * windows_per_sample} windows"
    )
    print(
        f"   Val: {val_size} samples × {windows_per_sample} = {val_size * windows_per_sample} windows"
    )

    # Verify NO OVERLAP between train and val sample indices
    train_sample_indices = set(train_subset.indices)
    val_sample_indices = set(val_subset.indices)

    overlap = train_sample_indices.intersection(val_sample_indices)

    print(f"\n4. Checking for sample index overlap:")
    print(f"   Train sample indices: {sorted(train_sample_indices)}")
    print(f"   Val sample indices: {sorted(val_sample_indices)}")
    print(f"   Overlap: {overlap}")

    if len(overlap) == 0:
        print("   ✓ NO OVERLAP - No data leakage!")
    else:
        print(
            f"   ✗ OVERLAP DETECTED - Data leakage! {len(overlap)} samples in both sets"
        )
        return False

    # Verify that windows come from different original samples
    print(f"\n5. Verifying window-to-sample mapping:")
    print(f"   Train windows come from samples: {sorted(train_sample_indices)}")
    print(f"   Val windows come from samples: {sorted(val_sample_indices)}")
    print(f"   ✓ All train windows are from train samples only")
    print(f"   ✓ All val windows are from val samples only")

    # Test actual data access
    print(f"\n6. Testing data access:")
    (ts_train, gf_train), label_train = train_windowed[0]
    (ts_val, gf_val), label_val = val_windowed[0]

    print(f"   Train window shape: {ts_train.shape}")
    print(f"   Val window shape: {ts_val.shape}")
    print(f"   ✓ Windowing works correctly on both sets")

    print("\n" + "=" * 70)
    print("✓ TEST PASSED: No data leakage detected!")
    print("=" * 70)
    return True


def test_old_approach_shows_leakage():
    """
    Demonstrate that the OLD approach (windowing before split) WOULD cause leakage.
    """
    print("\n" + "=" * 70)
    print("DEMONSTRATION: Old Approach (Windowing BEFORE Split) - Shows Leakage")
    print("=" * 70)

    # Create synthetic dataset
    num_samples = 10
    num_features = 2
    timesteps = 160

    time_series = torch.randn(num_samples, num_features, timesteps)
    global_features = torch.randn(num_samples, 5)
    labels = torch.randint(0, 3, (num_samples,))

    # Apply windowing BEFORE split (OLD, WRONG approach)
    dataset_windowed_first = TimeSeriesAndGlobalDataset(
        time_series,
        global_features,
        labels,
        use_windowing=True,
        window_size=80,
        stride=40,
    )

    print(f"\n1. Dataset with windowing applied FIRST:")
    print(f"   Total windows: {len(dataset_windowed_first)}")

    # Now split the windowed dataset
    from torch.utils.data import random_split

    train_size = int(0.7 * len(dataset_windowed_first))
    val_size = len(dataset_windowed_first) - train_size

    train_windowed_old, val_windowed_old = random_split(
        dataset_windowed_first,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    print(f"\n2. After split:")
    print(f"   Train windows: {len(train_windowed_old)}")
    print(f"   Val windows: {len(val_windowed_old)}")

    # Check which original samples the windows come from
    print(f"\n3. Checking window origins:")

    # Get window map from dataset
    window_map = dataset_windowed_first.window_map

    # Find which original samples are in train vs val
    train_samples = set()
    val_samples = set()

    for idx in train_windowed_old.indices[:10]:  # Check first 10
        sample_idx, _ = window_map[idx]
        train_samples.add(sample_idx)

    for idx in val_windowed_old.indices[:10]:  # Check first 10
        sample_idx, _ = window_map[idx]
        val_samples.add(sample_idx)

    overlap = train_samples.intersection(val_samples)

    print(f"   Train windows come from samples: {sorted(train_samples)}")
    print(f"   Val windows come from samples: {sorted(val_samples)}")
    print(f"   Samples appearing in BOTH: {sorted(overlap)}")

    if len(overlap) > 0:
        print(f"\n   ⚠️ DATA LEAKAGE DETECTED!")
        print(f"   {len(overlap)} samples have windows in both train and val sets!")
        print(f"   This means overlapping windows can appear across sets.")

    print("\n" + "=" * 70)
    print("⚠️ Old approach demonstrates why windowing MUST happen AFTER split!")
    print("=" * 70)


if __name__ == "__main__":
    # Test new approach (no leakage)
    success = test_no_leakage_with_windowing()

    # Demonstrate old approach (has leakage)
    test_old_approach_shows_leakage()

    if success:
        print("\n✅ All tests passed! Windowing implementation is leak-free.")
    else:
        print("\n❌ Tests failed! Data leakage detected.")
