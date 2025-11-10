import pytest
import pandas as pd
import yaml
from src.GradientGang.PreProcessing.PreProcessor import PreProcessor


def test_labelsWeights():
    test_dataframe_label = pd.DataFrame({
        'label': ['A', 'B', 'A', 'C', 'B', 'A']
    })
    preprocessor = PreProcessor({
        'data_path': './',
        'batch_size': 2,
        'num_workers': 0
    })
    preprocessor.computeAndSaveClassWeights(
        test_dataframe_label, 'test_weights.yaml')
    with open('./test_weights.yaml', 'r') as file:
        labels_weights = yaml.safe_load(file)

    # Formula: weight = total_samples / (num_classes * count_for_class)
    # total_samples = 6, num_classes = 3
    # A: 6 / (3 * 3) = 6/9 = 0.6667
    # B: 6 / (3 * 2) = 6/6 = 1.0
    # C: 6 / (3 * 1) = 6/3 = 2.0
    expected_weights = {
        'A': 6 / (3 * 3),  # 3 occurrences -> weight = 0.6667
        'B': 6 / (3 * 2),  # 2 occurrences -> weight = 1.0
        'C': 6 / (3 * 1)   # 1 occurrence -> weight = 2.0
    }

    # Check that computed weights match expected
    for label, expected_weight in expected_weights.items():
        assert label in labels_weights, f"Label {label} not found in computed weights"
        assert abs(labels_weights[label] - expected_weight) < 1e-6, \
            f"Weight for {label}: expected {expected_weight}, got {labels_weights[label]}"
