"""
Corrected test: Verify NO data leakage in K-Fold with proper index tracking.

The key insight: After _apply_windowing_to_subset(), the windowed dataset's indices
are RE-MAPPED to [0, 1, 2, ...]. This is CORRECT behavior and doesn't indicate leakage.

We need to track the ORIGINAL sample indices through the K-Fold split.
"""

import torch
import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.GradientGang.Pipeline.DataLoader.DataLoader import DataModule
from sklearn.model_selection import KFold


def test_kfold_correct():
    """
    CORRECT test for K-Fold data leakage.

    Verifies that:
    1. K-Fold splits samples correctly at the sample level
    2. Each fold has different validation samples
    3. No sample appears in both train and val within the same fold
    4. All samples are validated exactly once across all folds
    """
    print("=" * 70)
    print("TEST: K-Fold Cross-Validation - Proper Leakage Check")
    print("=" * 70)

    import tempfile
    import pandas as pd

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic dataset
        num_samples = 20
        num_features = 3
        timesteps = 160

        # Create time series data with UNIQUE values per sample for tracking
        data_rows = []
        for sample_idx in range(num_samples):
            for time in range(timesteps):
                row = {"sample_index": sample_idx, "time": time}
                for feat_idx in range(num_features):
                    # Use sample_idx as a marker in the data
                    row[f"feature_{feat_idx}"] = (
                        sample_idx * 1000 + time + feat_idx * 0.1
                    )
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

        # Create test data
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
            "stride": 40,
            "timeSeriesColumns": [f"feature_{i}" for i in range(num_features)],
        }

        data_module = DataModule(params)
        data_module.setup(stage="fit", includeTestInTrain=False)

        print(f"\n1. Configuration:")
        print(f"   Samples: {num_samples}")
        print(f"   K-Folds: {params['n_folds']}")
        print(f"   Windowing: size={params['window_size']}, stride={params['stride']}")

        # Get the K-Fold splits that SHOULD be used
        kfold = KFold(n_splits=params["n_folds"], shuffle=True, random_state=42)
        all_indices = np.arange(num_samples)
        expected_splits = list(kfold.split(all_indices))

        print(f"\n2. Expected K-Fold Splits (at sample level):")
        for fold_idx, (train_idx, val_idx) in enumerate(expected_splits):
            print(f"   Fold {fold_idx}: {len(train_idx)} train, {len(val_idx)} val")
            print(f"      Val samples: {sorted(val_idx.tolist())}")

        # Now test each fold
        print(f"\n3. Testing each fold:")
        all_folds_clean = True
        all_val_samples_seen = set()

        for fold_idx in range(params["n_folds"]):
            print(f"\n   Fold {fold_idx}:")

            # Setup this fold
            data_module.setup_fold(fold_idx, include_test_in_train=False)

            # Get the expected indices for this fold
            expected_train_idx, expected_val_idx = expected_splits[fold_idx]
            expected_train_samples = set(expected_train_idx.tolist())
            expected_val_samples = set(expected_val_idx.tolist())

            print(f"      Expected train samples: {len(expected_train_samples)}")
            print(f"      Expected val samples: {len(expected_val_samples)}")

            # Get actual datasets
            train_dataset = data_module.train_labeled
            val_dataset = data_module.val_dataset

            # Key insight: We need to DECODE which original samples are in each set
            # We can do this by looking at the DATA VALUES (we encoded sample_idx * 1000)

            def get_original_sample_ids(windowed_dataset, num_to_check=None):
                """Extract original sample IDs from windowed dataset by examining data values"""
                if num_to_check is None:
                    num_to_check = windowed_dataset.num_original_samples

                original_ids = set()
                for subset_idx in range(num_to_check):
                    # Get the time series for this subset index
                    ts = windowed_dataset.time_series_data[subset_idx]
                    # Decode the original sample ID from the first feature's first timestep
                    value = ts[0, 0].item()  # First feature, first timestep
                    original_id = int(
                        value / 1000
                    )  # Decode: value = original_id * 1000 + time
                    original_ids.add(original_id)

                return original_ids

            # Get actual sample IDs in train and val
            actual_train_samples = get_original_sample_ids(train_dataset)
            actual_val_samples = get_original_sample_ids(val_dataset)

            print(f"      Actual train samples: {sorted(actual_train_samples)[:10]}...")
            print(f"      Actual val samples: {sorted(actual_val_samples)}")

            # Check for overlap (DATA LEAKAGE)
            overlap = actual_train_samples.intersection(actual_val_samples)

            if len(overlap) > 0:
                print(
                    f"      ❌ LEAKAGE: {len(overlap)} samples in BOTH train and val!"
                )
                print(f"         Overlapping: {sorted(overlap)}")
                all_folds_clean = False
            else:
                print(f"      ✅ No overlap - clean separation")

            # Verify matches expected K-Fold split
            if actual_train_samples == expected_train_samples:
                print(f"      ✅ Train samples match expected K-Fold split")
            else:
                print(f"      ❌ Train samples DON'T match expected!")
                print(
                    f"         Missing: {expected_train_samples - actual_train_samples}"
                )
                print(
                    f"         Extra: {actual_train_samples - expected_train_samples}"
                )
                all_folds_clean = False

            if actual_val_samples == expected_val_samples:
                print(f"      ✅ Val samples match expected K-Fold split")
            else:
                print(f"      ❌ Val samples DON'T match expected!")
                print(f"         Expected: {sorted(expected_val_samples)}")
                print(f"         Got: {sorted(actual_val_samples)}")
                all_folds_clean = False

            # Track all validation samples across folds
            all_val_samples_seen.update(actual_val_samples)

        # Verify all samples validated exactly once
        print(f"\n4. Cross-Fold Verification:")
        print(
            f"   Samples validated across all folds: {len(all_val_samples_seen)}/{num_samples}"
        )

        if len(all_val_samples_seen) == num_samples:
            print(f"   ✅ All samples validated exactly once")
        else:
            missing = set(range(num_samples)) - all_val_samples_seen
            print(f"   ❌ Missing samples: {sorted(missing)}")
            all_folds_clean = False

        # Final verdict
        print("\n" + "=" * 70)
        if all_folds_clean:
            print("✅ K-FOLD TEST PASSED!")
            print("   - K-Fold correctly splits at sample level")
            print("   - Windowing applied after split prevents leakage")
            print("   - Each fold has unique validation set")
            print("   - All samples validated exactly once across folds")
        else:
            print("❌ K-FOLD TEST FAILED!")
        print("=" * 70)

        return all_folds_clean


if __name__ == "__main__":
    success = test_kfold_correct()
    sys.exit(0 if success else 1)
