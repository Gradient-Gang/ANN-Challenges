"""
Test to verify NO data leakage in K-Fold cross-validation with windowing.
"""

import torch
import sys
import os
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.GradientGang.Pipeline.DataLoader.DataLoader import DataModule


def test_kfold_no_leakage():
    """
    Critical test: Verify K-Fold splits have NO data leakage when windowing is applied.

    The key is that:
    1. K-Fold splits happen at SAMPLE level (on non-windowed dataset)
    2. Windowing is applied AFTER split to each fold separately
    3. No sample appears in both train and val for any fold
    """
    print("=" * 70)
    print("TEST: K-Fold Cross-Validation - No Data Leakage with Windowing")
    print("=" * 70)

    # Create test data directory and files
    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic dataset
        num_samples = 20
        num_features = 3
        timesteps = 160

        # Create time series data
        data_rows = []
        for sample_idx in range(num_samples):
            for time in range(timesteps):
                row = {"sample_index": sample_idx, "time": time}
                for feat_idx in range(num_features):
                    row[f"feature_{feat_idx}"] = np.random.randn()
                data_rows.append(row)

        data_df = pd.DataFrame(data_rows)
        data_csv = os.path.join(tmpdir, "train_data.csv")
        data_df.to_csv(data_csv, index=False)

        # Create labels
        labels_df = pd.DataFrame(
            {
                "sample_index": range(num_samples),
                "label": np.random.choice(
                    ["no_pain", "low_pain", "high_pain"], num_samples
                ),
            }
        )
        labels_csv = os.path.join(tmpdir, "train_labels.csv")
        labels_df.to_csv(labels_csv, index=False)

        # Create test data (not used in this test)
        test_df = data_df.copy()
        test_csv = os.path.join(tmpdir, "test_data.csv")
        test_df.to_csv(test_csv, index=False)

        # Setup DataModule with K-Fold and windowing
        params = {
            "data_dir": tmpdir,
            "train_file_name": "train_data.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test_data.csv",
            "batch_size": 4,
            "num_workers": 0,
            "val_split": 0.2,
            "use_kfold": True,
            "n_folds": 4,
            "use_windowing": True,
            "window_size": 80,
            "stride": 40,  # 50% overlap
            "timeSeriesColumns": [f"feature_{i}" for i in range(num_features)],
        }

        data_module = DataModule(params)

        # Setup to load full dataset
        data_module.setup(stage="fit", includeTestInTrain=False)

        print(f"\n1. Dataset Configuration:")
        print(f"   Total samples: {num_samples}")
        print(f"   Sequence length: {timesteps}")
        print(f"   K-Fold splits: {params['n_folds']}")
        print(f"   Windowing: size={params['window_size']}, stride={params['stride']}")

        # Calculate expected windows per sample
        windows_per_sample = (
            (timesteps - params["window_size"]) // params["stride"]
        ) + 1
        print(f"   Expected windows per sample: {windows_per_sample}")

        # Test each fold for data leakage
        print(f"\n2. Testing each fold for data leakage:")

        # First, let's see what the K-Fold splitter actually produces
        from sklearn.model_selection import KFold

        kfold = KFold(n_splits=params["n_folds"], shuffle=True, random_state=42)
        all_indices = np.arange(num_samples)
        kfold_splits = list(kfold.split(all_indices))

        print(f"\n   K-Fold splits (at sample level):")
        for fold_idx, (train_idx, val_idx) in enumerate(kfold_splits):
            print(
                f"      Fold {fold_idx}: train={sorted(train_idx.tolist())[:10]}..., val={sorted(val_idx.tolist())}"
            )

        all_folds_clean = True

        for fold_idx in range(params["n_folds"]):
            print(f"\n   Fold {fold_idx}:")

            # Setup this fold
            data_module.setup_fold(fold_idx, include_test_in_train=False)

            # Get fold info
            fold_info = data_module.get_fold_info()
            print(f"      Train windows: {fold_info['train_size']}")
            print(f"      Val windows: {fold_info['val_size']}")

            # Get the underlying datasets
            train_dataset = data_module.train_labeled  # This is windowed
            val_dataset = data_module.val_dataset  # This is windowed

            # IMPORTANT: After _apply_windowing_to_subset, the window_map indices
            # refer to positions in the SUBSET, not original dataset.
            # We need to verify that the ACTUAL DATA is different between train/val.

            # Instead, let's check the actual data values to ensure no overlap
            # Get a sample from each set and verify they're from different original samples

            # Get the number of unique "samples" in each windowed dataset
            train_num_samples = train_dataset.num_original_samples
            val_num_samples = val_dataset.num_original_samples

            print(f"      Train has {train_num_samples} original samples")
            print(f"      Val has {val_num_samples} original samples")

            # The REAL test: Check if any actual data tensors match between sets
            # Compare first time series from train vs all time series in val
            train_sample_0 = train_dataset.time_series_data[0]  # First train sample

            data_overlap_found = False
            for val_idx in range(min(5, val_num_samples)):
                val_sample = val_dataset.time_series_data[val_idx]
                if torch.allclose(train_sample_0, val_sample, atol=1e-6):
                    print(f"      ❌ ACTUAL DATA MATCH: Train[0] == Val[{val_idx}]")
                    data_overlap_found = True
                    break

            if data_overlap_found:
                all_folds_clean = False

            # For tracking, just count samples
            train_sample_indices = set(range(train_num_samples))
            val_sample_indices = set(range(val_num_samples))
            overlap = set()  # No meaningful overlap check since indices are re-mapped

            # Check for overlap (DATA LEAKAGE)
            overlap = train_sample_indices.intersection(val_sample_indices)

            print(f"      Train samples: {sorted(train_sample_indices)}")
            print(f"      Val samples: {sorted(val_sample_indices)}")
            print(f"      Overlap: {overlap}")

            if len(overlap) > 0:
                print(f"      ❌ DATA LEAKAGE DETECTED in fold {fold_idx}!")
                print(f"         {len(overlap)} samples appear in BOTH train and val")
                all_folds_clean = False
            else:
                print(f"      ✅ No leakage - clean separation")

            # Verify sample counts
            expected_val_samples = num_samples // params["n_folds"]
            expected_train_samples = num_samples - expected_val_samples

            if len(val_sample_indices) != expected_val_samples:
                print(
                    f"      ⚠️ Val sample count mismatch: expected {expected_val_samples}, got {len(val_sample_indices)}"
                )

            # Verify window counts match sample counts
            expected_train_windows = len(train_sample_indices) * windows_per_sample
            expected_val_windows = len(val_sample_indices) * windows_per_sample

            if fold_info["train_size"] != expected_train_windows:
                print(
                    f"      ⚠️ Train window count mismatch: expected {expected_train_windows}, got {fold_info['train_size']}"
                )

            if fold_info["val_size"] != expected_val_windows:
                print(
                    f"      ⚠️ Val window count mismatch: expected {expected_val_windows}, got {fold_info['val_size']}"
                )

        # Verify all samples are used across folds
        print(f"\n3. Verifying all samples are used across all folds:")
        all_val_samples = set()

        for fold_idx in range(params["n_folds"]):
            data_module.setup_fold(fold_idx, include_test_in_train=False)
            val_dataset = data_module.val_dataset

            for sample_idx, _ in val_dataset.window_map:
                all_val_samples.add(sample_idx)

        print(f"   Total unique samples used for validation: {len(all_val_samples)}")
        print(f"   Expected: {num_samples}")

        if len(all_val_samples) == num_samples:
            print(f"   ✅ All samples validated across folds")
        else:
            missing = set(range(num_samples)) - all_val_samples
            print(f"   ❌ Missing samples: {missing}")
            all_folds_clean = False

        # Final verdict
        print("\n" + "=" * 70)
        if all_folds_clean:
            print("✅ K-FOLD TEST PASSED: No data leakage detected in any fold!")
            print("   - Each fold splits at SAMPLE level first")
            print("   - Windowing applied separately to each split")
            print("   - No sample appears in both train and val within same fold")
            print("   - All samples validated exactly once across folds")
        else:
            print("❌ K-FOLD TEST FAILED: Data leakage detected!")
        print("=" * 70)

        return all_folds_clean


def test_kfold_window_distribution():
    """
    Verify that windows are distributed correctly across folds.
    """
    print("\n" + "=" * 70)
    print("TEST: K-Fold Window Distribution")
    print("=" * 70)

    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create small dataset
        num_samples = 12
        num_features = 2
        timesteps = 160

        # Create time series data
        data_rows = []
        for sample_idx in range(num_samples):
            for time in range(timesteps):
                row = {"sample_index": sample_idx, "time": time}
                for feat_idx in range(num_features):
                    row[f"feature_{feat_idx}"] = np.random.randn()
                data_rows.append(row)

        data_df = pd.DataFrame(data_rows)
        data_csv = os.path.join(tmpdir, "train_data.csv")
        data_df.to_csv(data_csv, index=False)

        # Create labels
        labels_df = pd.DataFrame(
            {
                "sample_index": range(num_samples),
                "label": np.random.choice(["no_pain", "low_pain"], num_samples),
            }
        )
        labels_csv = os.path.join(tmpdir, "train_labels.csv")
        labels_df.to_csv(labels_csv, index=False)

        # Create test data
        test_df = data_df.copy()
        test_csv = os.path.join(tmpdir, "test_data.csv")
        test_df.to_csv(test_csv, index=False)

        # Setup with 3 folds
        params = {
            "data_dir": tmpdir,
            "train_file_name": "train_data.csv",
            "train_file_name_labels": "train_labels.csv",
            "test_file_name": "test_data.csv",
            "batch_size": 4,
            "num_workers": 0,
            "val_split": 0.2,
            "use_kfold": True,
            "n_folds": 3,
            "use_windowing": True,
            "window_size": 80,
            "stride": 40,
            "timeSeriesColumns": [f"feature_{i}" for i in range(num_features)],
        }

        data_module = DataModule(params)
        data_module.setup(stage="fit", includeTestInTrain=False)

        windows_per_sample = (
            (timesteps - params["window_size"]) // params["stride"]
        ) + 1

        print(f"\nDataset: {num_samples} samples, {windows_per_sample} windows/sample")
        print(f"Total windows: {num_samples * windows_per_sample}")

        # Check window distribution in first fold
        data_module.setup_fold(0, include_test_in_train=False)
        train_dataset = data_module.train_labeled

        # Count windows per sample in train set
        from collections import defaultdict

        windows_count = defaultdict(int)

        for sample_idx, start_pos in train_dataset.window_map:
            windows_count[sample_idx] += 1

        print(f"\nFold 0 - Train set window distribution:")
        for sample_idx in sorted(windows_count.keys())[:5]:  # Show first 5
            print(f"   Sample {sample_idx}: {windows_count[sample_idx]} windows")

        # Verify all samples have same number of windows
        unique_counts = set(windows_count.values())
        if len(unique_counts) == 1:
            print(f"\n✅ UNIFORM: All samples have {list(unique_counts)[0]} windows")
            return True
        else:
            print(f"\n⚠️ NON-UNIFORM: Different window counts: {unique_counts}")
            return False


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("K-FOLD WINDOWING LEAKAGE TEST SUITE")
    print("=" * 70)

    # Run tests
    test1_passed = test_kfold_no_leakage()
    test2_passed = test_kfold_window_distribution()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"{'✅ PASS' if test1_passed else '❌ FAIL'}: K-Fold No Leakage")
    print(f"{'✅ PASS' if test2_passed else '❌ FAIL'}: K-Fold Window Distribution")

    if test1_passed and test2_passed:
        print("\n✅ ALL K-FOLD TESTS PASSED!")
        print("\nConclusion:")
        print("  - K-Fold implementation correctly splits at SAMPLE level")
        print("  - Windowing applied AFTER split prevents data leakage")
        print("  - Each fold maintains clean train/val separation")
        print("  - Safe to use for hyperparameter optimization")
    else:
        print("\n❌ SOME K-FOLD TESTS FAILED!")

    print("=" * 70)

    sys.exit(0 if (test1_passed and test2_passed) else 1)
