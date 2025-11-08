import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset, random_split
import pandas as pd
import os
from ..Utils.ParameterInterpreter import ParameterInterpreter

DataLoaderInterpreter = ParameterInterpreter(
    name="DataLoaderInterpreter",
    interpretation={
        "data_dir": str,
        "train_file_name": str,
        "train_file_name_labels": str,
        "test_file_name": str,
        "batch_size": int,
        "num_workers": int,
        "val_split": float,
        "label_mapping": dict
    },
    requiredParams={
        "data_dir": str,
        "train_file_name": str,
        "train_file_name_labels": str,
        "test_file_name": str,
        "batch_size": int,
        "num_workers": int,
    }
)


class DataModule(L.LightningDataModule):
    """
    DataModule for loading datasets from CSV files and preparing PyTorch DataLoaders.
    Supports training, validation, and testing splits, as well as label mapping.
    """

    def __init__(self, params: dict):
        super().__init__()
        DataLoaderInterpreter.checkRequiredParams(params)

        self.data_dir = params.get("data_dir", "")
        self.train_file_name = params.get("train_file_name", "")
        self.train_file_name_labels = params.get("train_file_name_labels", "")
        self.test_file_name = params.get("test_file_name", "")
        self.batch_size = params.get("batch_size", 32)
        self.num_workers = params.get("num_workers", 0)
        self.val_split = params.get("val_split", 0.1)
        
        # Label mapping for converting string labels to integers
        # Default mapping for pirate pain dataset
        self.label_mapping = params.get("label_mapping", {
            "no_pain": 0,
            "low_pain": 1,
            "high_pain": 2
        })
        
        # Datasets will be initialized in setup()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def load_from_csv(self, file_path: str, label_file_path: str = None):
        """
        Load data from CSV files and convert to PyTorch tensors.
        """
        # Load features
        features_df = pd.read_csv(file_path)
        
        # Remove non-numeric columns if present (e.g., sample_index, time)
        numeric_cols = features_df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns
        features_df = features_df[numeric_cols]
        
        # Convert to tensor
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32)
        
        # Load labels if provided
        if label_file_path is not None:
            labels_df = pd.read_csv(label_file_path)
            
            # Check if labels need to be mapped from strings to integers
            if 'label' in labels_df.columns:
                if labels_df['label'].dtype == 'object':  # String labels
                    labels_df['label'] = labels_df['label'].map(self.label_mapping)
                labels_tensor = torch.tensor(labels_df['label'].values, dtype=torch.long)
            else:
                # Assume the first column after sample_index is the label
                label_col = labels_df.columns[-1] if 'sample_index' in labels_df.columns else labels_df.columns[0]
                labels_tensor = torch.tensor(labels_df[label_col].values, dtype=torch.long)
        else:
            labels_tensor = None
            
        return features_tensor, labels_tensor

    def setup(self, stage: str = None):
        """
        Setup datasets for training, validation, and testing.
        """
        if stage == 'fit' or stage is None:
            # Load training data and labels
            train_path = os.path.join(self.data_dir, self.train_file_name)
            train_labels_path = os.path.join(self.data_dir, self.train_file_name_labels)
            
            features, labels = self.load_from_csv(train_path, train_labels_path)
            
            # Create full training dataset
            full_dataset = TensorDataset(features, labels)
            
            # Split into train and validation
            val_size = int(len(full_dataset) * self.val_split)
            train_size = len(full_dataset) - val_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducibility
            )
            
        if stage == 'test' or stage is None:
            # Load test data (usually without labels)
            test_path = os.path.join(self.data_dir, self.test_file_name)
            
            features, labels = self.load_from_csv(test_path, None)
            
            # If no labels, create dummy labels for compatibility
            if labels is None:
                labels = torch.zeros(len(features), dtype=torch.long)
                
            self.test_dataset = TensorDataset(features, labels)

    def train_dataloader(self):
        """
        Create training dataloader.
        """
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")
            
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available()
        )

    def val_dataloader(self):
        """
        Create validation dataloader.
        """
        if self.val_dataset is None:
            raise RuntimeError("Validation dataset not initialized. Call setup() first.")
            
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available()
        )

    def test_dataloader(self):
        """
        Create test dataloader.
        """
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
            
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available()
        )

    def predict_dataloader(self):
        """
        Create prediction dataloader (uses test dataset).
        """
        return self.test_dataloader() 