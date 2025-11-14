"""Mock checkpoint generators for testing ensemble loading."""

import pytest
import torch
import os
from pathlib import Path


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Create temporary directory structure for checkpoints."""
    base_dir = tmp_path / "FinalPipelineLogs" / "Study_test_study"
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


@pytest.fixture
def create_mock_checkpoint(tmp_path, mock_checkpoint_dir):
    """Factory fixture for creating mock checkpoint files."""

    def _create_checkpoint(
        trial_number: int,
        fold_idx: int,
        val_f1: float,
        use_windowing: bool = False,
        model_state_dict=None,
    ):
        """Create a mock checkpoint file.

        Args:
            trial_number: Trial number
            fold_idx: Fold index
            val_f1: Validation F1 score
            use_windowing: Whether model uses windowing
            model_state_dict: Optional model state dict to save

        Returns:
            Path to created checkpoint file
        """
        # Create checkpoint directory
        ckpt_dir = mock_checkpoint_dir / f"version_{trial_number}" / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create checkpoint filename
        ckpt_path = (
            ckpt_dir / f"trial-{trial_number}-fold-{fold_idx}-val_F1={val_f1:.4f}.ckpt"
        )

        # Create mock model state dict if not provided
        if model_state_dict is None:
            model_state_dict = {
                "encoder.0.weight": torch.randn(64, 10),
                "encoder.0.bias": torch.randn(64),
                "classifier.weight": torch.randn(2, 64),
                "classifier.bias": torch.randn(2),
            }

        # Add windowing wrapper keys if needed
        if use_windowing:
            model_state_dict = {
                f"base_model.{k}": v for k, v in model_state_dict.items()
            }

        # Create checkpoint dictionary
        checkpoint = {
            "state_dict": model_state_dict,
            "epoch": 10,
            "global_step": 1000,
            "val_F1": val_f1,
        }

        # Save checkpoint
        torch.save(checkpoint, ckpt_path)

        return ckpt_path

    return _create_checkpoint


@pytest.fixture
def mock_fold_checkpoints(create_mock_checkpoint):
    """Create complete set of mock checkpoints for all folds."""

    def _create_fold_checkpoints(
        trial_number: int = 0,
        num_folds: int = 5,
        f1_scores=None,
        use_windowing: bool = False,
    ):
        """Create checkpoints for all folds.

        Args:
            trial_number: Trial number
            num_folds: Number of folds
            f1_scores: List of F1 scores (or None for default)
            use_windowing: Whether models use windowing

        Returns:
            List of checkpoint paths
        """
        if f1_scores is None:
            f1_scores = [0.90, 0.88, 0.89, 0.91, 0.87]

        checkpoints = []
        for fold_idx in range(num_folds):
            ckpt_path = create_mock_checkpoint(
                trial_number=trial_number,
                fold_idx=fold_idx,
                val_f1=f1_scores[fold_idx],
                use_windowing=use_windowing,
            )
            checkpoints.append(ckpt_path)

        return checkpoints

    return _create_fold_checkpoints


@pytest.fixture
def corrupted_checkpoint(mock_checkpoint_dir):
    """Create a corrupted checkpoint file."""
    ckpt_dir = mock_checkpoint_dir / "version_0" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "trial-0-fold-0-val_F1=0.9000.ckpt"

    # Write invalid data
    with open(ckpt_path, "wb") as f:
        f.write(b"corrupted data")

    return ckpt_path
