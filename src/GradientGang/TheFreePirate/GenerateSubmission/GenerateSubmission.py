from ..Architectures.PirateLightningModule import PirateLightningModule
from ..DataHandling.PirateDataModule import PirateDataModule, WindowingAugmenter
import torch
import pandas as pd
import os
import numpy as np


def generateSubmission(best_params, csv_path):
        dataParams = {
            'folderPath': '../../dataset/PirateProcessed',
            'trainTimeSeriesFileName': 'pirate_pain_train.csv',
            'trainGlobalFeaturesFileName': 'train_global_features.csv',
            'trainLabelsFileName': 'pirate_pain_train_labels.csv',
            'testTimeSeriesFileName': 'pirate_pain_test.csv',
            'testGlobalFeaturesFileName': 'test_global_features.csv',
            'labelsMapping': {'no_pain': 0, 'low_pain': 1, 'high_pain': 2},
            'batch_size': 32,
            'num_workers': 0
        }
        callbacksParams = {
            'baseLogDir': 'PirateLogs',
            'earlyStoppingPatience': 20,
            'maxEpochs': 50,
        }

        best_model_path = callbacksParams["baseLogDir"] + "\\test"

        numFolds = len(os.listdir(best_model_path))

        data = PirateDataModule.fromCSV(**dataParams)

        dataAug = best_params["dataParams"]["dataAugmentationParams"]

        testSet = data.getTestLoader(
            windowAugmenter=WindowingAugmenter(
                windowSize=dataAug["windowSize"],
                stride=dataAug["windowStride"]
            )
        )

        # Extract predictions
        all_predictions = []

        with torch.no_grad():
            for i in range(numFolds):
                # Load model for current fold
                fold_model_path = f"{best_model_path}\\fold_{i}"
                model = PirateLightningModule.load_from_checkpoint(fold_model_path)
                model.eval()
                
                fold_predictions = []
                
                for batch in testSet:
                    # Forward pass
                    if isinstance(batch, (list, tuple)):
                        features = batch[0]
                    else:
                        features = batch
                    
                    outputs = model(features)
                    
                    # Get predicted class indices
                    if isinstance(outputs, torch.Tensor):
                        predicted_classes = torch.argmax(outputs, dim=1)
                    elif isinstance(outputs, dict):
                        predicted_classes = torch.argmax(outputs['logits'], dim=1)
                    else:
                        raise ValueError("Unexpected model output format")
                    
                    fold_predictions.extend(predicted_classes.cpu().numpy())
                
                all_predictions.append(fold_predictions)

        # Convert list of fold predictions to numpy array for easier aggregation
        all_predictions = np.array(all_predictions)

        # Aggregate predictions across folds (majority voting)
        final_predictions = []
        for sample_idx in range(all_predictions.shape[1]):
            # Get predictions for this sample across all folds
            sample_predictions = all_predictions[:, sample_idx]
            # Use majority vote
            final_prediction = np.argmax(np.bincount(sample_predictions))
            final_predictions.append(final_prediction)

        # Convert numerical predictions to textual labels
        reverse_mapping = {0: 'no_pain', 1: 'low_pain', 2: 'high_pain'}
        textual_predictions = [reverse_mapping[pred] for pred in all_predictions]
        
        # Create sample indices with 3 precision digits (001, 002, 003, ...)
        sample_indices = [f"{i:03d}" for i in range(len(all_predictions))]
        
        # Create submission DataFrame in the required format
        submission_df = pd.DataFrame({
            'sample_index': sample_indices,
            'label': textual_predictions
        })
        
        # Save to CSV with the exact header format
        submission_df.to_csv(csv_path, index=False)
        print(f"Submission saved to {csv_path}")
        print(f"Generated {len(submission_df)} predictions")