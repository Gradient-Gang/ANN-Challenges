import pytorch_lightning as L
import torch
from torch.utils.data import (
    DataLoader as TorchDataLoader,
    Dataset,
    random_split,
    Subset,
)
import pandas as pd
import os
import numpy as np
from torch.utils.data.dataset import ConcatDataset
from sklearn.model_selection import StratifiedKFold
from ..Utils.ParameterInterpreter import ParameterInterpreter
from .Augmentations import AugmentationPipeline


class SubsetWithTrainingFlag(Subset):
    """
    A Subset wrapper that overrides the is_training flag on the underlying dataset
    when accessing items. This allows train and val subsets from the same dataset
    to have different augmentation behavior.
    """
    def __init__(self, dataset, indices, is_training: bool):
        super().__init__(dataset, indices)
        self._is_training = is_training
        
    def __getitem__(self, idx):
        # Temporarily override is_training flag
        original_is_training = self.dataset.is_training
        self.dataset.is_training = self._is_training
        try:
            result = super().__getitem__(idx)
        finally:
            # Restore original flag
            self.dataset.is_training = original_is_training
        return result


class TimeSeriesAndGlobalDataset(Dataset):

    @staticmethod
    def fromCSV(
        dataPath: str,
        labelsPath: str | None = None,
        primaryKeyColumn: str = "sample_index",
        globalColumns: list[str] = [],
        timeSeriesColumns: list[str] | None = None,
        labelMapping: list[str] = ["no_pain", "low_pain", "high_pain"],
        globalFeaturesPath: str | None = None,
        use_windowing: bool = False,
        window_size: int = 160,
        stride: int = 160,
        drop_column_list: list[str] = [],
        augmentation_pipeline: AugmentationPipeline | None = None,
        is_training: bool = True,
    ) -> "TimeSeriesAndGlobalDataset":
        """
        Create a TimeSeriesAndGlobalDataset from CSV files.
        Args:
            dataPath (str): Path to the CSV file containing the data.
            labelsPath (str | None): Path to the CSV file containing the labels. If None, no labels are loaded.
            primaryKeyColumn (str): Column name for the primary key.
            globalColumns (list[str]): List of column names for global features.
            timeSeriesColumns (list[str] | None): List of column names for time series features
            labelMapping (list[str]): List of possible labels for mapping string labels to integers.
            globalFeaturesPath (str | None): Path to CSV file containing pre-extracted global features (e.g., from PreProcessor)
            use_windowing (bool): Whether to apply windowing augmentation.
            window_size (int): Size of each window (default: 160, full sequence).
            stride (int): Stride for sliding window (default: 160, no overlap).
            drop_column_list (list[str]): List of column names to drop before building the dataloader.
        Returns:
            TimeSeriesAndGlobalDataset: The constructed dataset.
        """

        # Load data from CSV and check for required columns
        data_df = pd.read_csv(dataPath)

        # Drop specified columns if any
        if drop_column_list:
            columns_to_drop = [
                col for col in drop_column_list if col in data_df.columns
            ]
            if columns_to_drop:
                data_df = data_df.drop(columns=columns_to_drop)

        # Flag to track if global features were loaded from separate file
        global_features_loaded_separately = False

        # If separate global features file is provided, load and merge it
        if globalFeaturesPath is not None:
            try:
                global_features_df = pd.read_csv(globalFeaturesPath)

                # Drop specified columns from global features if any
                if drop_column_list:
                    columns_to_drop = [
                        col
                        for col in drop_column_list
                        if col in global_features_df.columns
                    ]
                    if columns_to_drop:
                        global_features_df = global_features_df.drop(
                            columns=columns_to_drop
                        )

                # Extract sample indices in order from time series data
                sample_ids = data_df[primaryKeyColumn].unique()
                # Reindex global features to match time series sample order
                global_features_df = (
                    global_features_df.set_index(primaryKeyColumn)
                    .reindex(sample_ids)
                    .reset_index()
                )
                # All columns except sample_index are global features
                globalColumns = [
                    col for col in global_features_df.columns if col != primaryKeyColumn
                ]
                globalFeatures = global_features_df[globalColumns].to_numpy()
                globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
                global_features_loaded_separately = True
            except Exception as e:
                print(
                    f"Warning: Could not load global features from {globalFeaturesPath}: {e}"
                )
                print(
                    "Falling back to extracting global features from time series data."
                )
                globalColumns = data_df.columns.intersection(globalColumns).tolist()
                globalFeatures = (
                    data_df.groupby(primaryKeyColumn).first()[globalColumns].to_numpy()
                )
                globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
        else:
            # Extract global features from time series data (original behavior)
            globalColumns = data_df.columns.intersection(globalColumns).tolist()
            globalFeatures = (
                data_df.groupby(primaryKeyColumn).first()[globalColumns].to_numpy()
            )
            globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)

        if timeSeriesColumns is None:
            timeSeriesColumns = data_df.columns.difference(
                [primaryKeyColumn, "time"] + globalColumns
            ).tolist()

        # Build an ordered list of sample primary keys and features
        if "time" in data_df.columns:
            # Use groupby first to obtain one row per sample and capture the sample order
            grouped = data_df.groupby(primaryKeyColumn).first()
            sample_ids = grouped.index.to_numpy()

            # Only extract global features from time series if they weren't loaded separately
            if not global_features_loaded_separately:
                globalFeatures = grouped[globalColumns].to_numpy()
                globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)

            # Process time series data
            if timeSeriesColumns is None:
                timeSeriesColumns = data_df.columns.difference(
                    [primaryKeyColumn, "time"] + globalColumns
                ).tolist()

            # Pivot and stack time series data
            timeSeriesDF = data_df[[primaryKeyColumn, "time"] + timeSeriesColumns]
            # Pivot each feature and reindex to ensure the same sample order as grouped
            timeSeries_list = []
            for feat in timeSeriesColumns:
                pivoted = timeSeriesDF.pivot(
                    index=primaryKeyColumn, columns="time", values=feat
                ).reindex(sample_ids)
                timeSeries_list.append(pivoted.to_numpy())

            timeSeries = np.stack(timeSeries_list, axis=-1)
            timeSeries = torch.tensor(timeSeries, dtype=torch.float32)
            timeSeries = timeSeries.permute(0, 2, 1)  # (samples, features, time)

        # No time series data
        else:
            # No time-series dimension; preserve row order
            if primaryKeyColumn in data_df.columns:
                sample_ids = data_df[primaryKeyColumn].to_numpy()
            else:
                sample_ids = np.arange(len(data_df))

            globalFeatures = data_df[globalColumns].to_numpy()
            globalFeatures = torch.tensor(globalFeatures, dtype=torch.float32)
            timeSeries = None

        # Load and align labels to the sample order (use -1 for unlabeled)
        if labelsPath is not None:
            labels_df = pd.read_csv(labelsPath)
            # Map textual labels to integers
            label_map = {label: idx for idx, label in enumerate(labelMapping)}

            if primaryKeyColumn in labels_df.columns:
                labels_series = labels_df.set_index(primaryKeyColumn)["label"].map(
                    label_map
                )
            else:
                # If no primary key in labels file, assume same order as samples
                labels_series = labels_df["label"].map(label_map)

            # Align labels to sample_ids and fill missing with -1
            labels_aligned = (
                pd.Series(sample_ids).map(labels_series).fillna(-1).astype(int)
            )
            labels = torch.tensor(labels_aligned.to_numpy(), dtype=torch.long)
        else:
            labels = torch.full((len(sample_ids),), -1, dtype=torch.long)

        # Return the constructed dataset
        return TimeSeriesAndGlobalDataset(
            timeSeries, globalFeatures, labels, use_windowing, window_size, stride,
            augmentation_pipeline, is_training
        )

    def __init__(
        self,
        time_series_data: torch.Tensor | None,
        global_data: torch.Tensor,
        labels: torch.Tensor | None,
        use_windowing: bool = False,
        window_size: int = 160,
        stride: int = 160,
        augmentation_pipeline: AugmentationPipeline | None = None,
        is_training: bool = True,
    ):
        """
        Initialize the TimeSeriesAndGlobalDataset.
        Args:
            time_series_data (torch.Tensor | None): Time series data tensor or None if not available.
            global_data (torch.Tensor): Global features tensor.
            labels (torch.Tensor | None): Labels tensor or None if not available.
            use_windowing (bool): Whether to apply windowing augmentation.
            window_size (int): Size of each window (default: 160, full sequence).
            stride (int): Stride for sliding window (default: 160, no overlap).
        """

        # Validate input dimensions (fail fast)
        if time_series_data is not None:
            assert len(time_series_data) == len(
                global_data
            ), "All inputs must have the same number of samples."

        if labels is not None:
            assert len(global_data) == len(
                labels
            ), "All inputs must have the same number of samples."

        # Store data
        self.time_series_data = time_series_data
        self.global_data = global_data
        self.labels = labels

        # Cache original number of samples (before windowing)
        self.num_original_samples = len(global_data)

        # Windowing parameters
        self.use_windowing = use_windowing
        self.window_size = window_size
        self.stride = stride
        
        # Augmentation parameters
        self.augmentation_pipeline = augmentation_pipeline
        self.is_training = is_training

        # Build window index if windowing is enabled
        self.window_map = []
        if use_windowing and time_series_data is not None:
            self._build_window_map()

        # Store dataset information for quick access (pre-computed)
        self.timeSeriesShape = (
            time_series_data.shape[1:] if time_series_data is not None else None
        )
        self.globalFeaturesShape = global_data.shape[1:]

        # Compute numClasses efficiently (avoid repeated .max() calls)
        if labels is not None and len(labels) > 0:
            # Use labels.unique() which is more efficient than max() for this purpose
            unique_labels = labels[labels >= 0].unique()  # Exclude -1 (unlabeled)
            self.numClasses = len(unique_labels) if len(unique_labels) > 0 else 0
        else:
            self.numClasses = 0

    def _build_window_map(self):
        """Build mapping: window_idx -> (sample_idx, start_pos) efficiently"""
        num_samples = len(self.time_series_data)
        seq_len = self.time_series_data.shape[2]  # Same for all samples

        # Pre-calculate number of windows per sample for efficient allocation
        num_windows_per_sample = ((seq_len - self.window_size) // self.stride) + 1

        # Generate all window positions for one sample (reusable pattern)
        window_starts = list(range(0, seq_len, self.stride))
        # Remove positions where window extends too far beyond sequence
        window_starts = [s for s in window_starts if s < seq_len]

        # Build map efficiently using list comprehension
        self.window_map = [
            (sample_idx, start)
            for sample_idx in range(num_samples)
            for start in window_starts
        ]

    def __len__(self):
        # Return cached length (windows if enabled, otherwise original samples)
        # Using window_map length is O(1) since it's pre-computed
        if self.use_windowing and self.window_map:
            return len(self.window_map)
        return self.num_original_samples

    def __getitem__(self, index):
        # If windowing is enabled, extract the appropriate window
        if self.use_windowing and len(self.window_map) > 0:
            sample_idx, start_pos = self.window_map[index]

            # Extract windowed time series (optimized with pre-computed end position)
            if self.time_series_data is not None:
                end_pos = start_pos + self.window_size

                # Fast path: no padding needed (most common case)
                if end_pos <= self.time_series_data.shape[2]:
                    windowed_ts = self.time_series_data[
                        sample_idx, :, start_pos:end_pos
                    ]
                else:
                    # Slow path: padding needed (rare)
                    actual_end = self.time_series_data.shape[2]
                    windowed_ts = self.time_series_data[
                        sample_idx, :, start_pos:actual_end
                    ]

                    # Use F.pad for efficiency instead of torch.cat
                    pad_size = self.window_size - windowed_ts.shape[1]
                    windowed_ts = torch.nn.functional.pad(
                        windowed_ts, (0, pad_size), mode="constant", value=0
                    )
                
                # Apply augmentation to windowed time series (training only)
                if self.is_training and self.augmentation_pipeline is not None:
                    windowed_ts = self.augmentation_pipeline.apply(windowed_ts, index)
            else:
                windowed_ts = None

            # Return windowed data with corresponding global features and label
            return (
                windowed_ts,
                self.global_data[sample_idx],
            ), (self.labels[sample_idx] if self.labels is not None else None)

        # Original behavior: retrieve the full sample at the specified index
        time_series = self.time_series_data[index] if self.time_series_data is not None else None
        
        # Apply augmentation to full time series (training only)
        if self.is_training and self.augmentation_pipeline is not None and time_series is not None:
            time_series = self.augmentation_pipeline.apply(time_series, index)
        
        return (
            time_series,
            self.global_data[index],
        ), (self.labels[index] if self.labels is not None else None)

    def __add__(self, other: Dataset) -> ConcatDataset:
        return ConcatDataset([self, other])


class DataModule(L.LightningDataModule):
    """
    DataModule for loading datasets from CSV files and preparing PyTorch DataLoaders.
    Supports training, validation, and testing splits, as well as label mapping.
    """

    # Parameter interpreter for validating input parameters
    DataLoaderInterpreter = ParameterInterpreter(
        name="DataLoaderInterpreter",
        interpretation={},
        requiredParams={
            "data_dir": str,
            "train_file_name": str,
            "train_file_name_labels": str,
            "test_file_name": str,
            "batch_size": int,
            "num_workers": int,
            "data_dir": str,
            "val_split": float,
        },
    )

    def __init__(self, params: dict):
        """
        Initialize the DataModule with parameters.
        Args:
            params (dict): Dictionary of parameters for DataModule configuration.
        """

        super().__init__()
        self.DataLoaderInterpreter.checkRequiredParams(params)

        # Initialize attributes from parameters
        self.data_dir = params.get("data_dir", "")
        self.train_file_name = params.get("train_file_name", "")
        self.train_file_name_labels = params.get("train_file_name_labels", "")
        self.test_file_name = params.get("test_file_name", "")
        self.batch_size = params.get("batch_size", 32)
        self.num_workers = params.get("num_workers", 0)
        self.val_split = params.get("val_split", 0.1)

        # Initialize feature columns and primary key
        self.globalFeaturesColumns = params.get(
            "globalFeaturesColumns", ["isPirate", "isNotPirate"]
        )
        self.primaryKeyColumn = params.get("primaryKeyColumn", "sample_index")
        self.timeSeriesColumns = params.get("timeSeriesColumns", None)

        # Optional: Separate file for global features
        self.train_global_features_file = params.get("train_global_features_file", None)
        self.test_global_features_file = params.get("test_global_features_file", None)

        # Label mapping for converting string labels to integers
        # Default mapping for pirate pain dataset
        self.label_mapping = params.get(
            "label_mapping", {"no_pain": 0, "low_pain": 1, "high_pain": 2}
        )

        # Random seed for train/val split (can be overridden via params)
        split_seed = params.get("split_seed", 42)
        self.trainValGenerator = torch.Generator().manual_seed(split_seed)

        # K-Fold Cross-Validation settings
        self.use_kfold = params.get("use_kfold", False)
        self.n_folds = params.get("n_folds", 5)
        self.current_fold = None
        self._full_labeled_dataset = None  # Store full dataset for K-Fold splitting

        # Windowing parameters
        self.use_windowing = params.get("use_windowing", False)
        self.window_size = params.get("window_size", 160)
        self.stride = params.get("stride", 160)
        
        # Augmentation parameters
        self.augmentation_config = params.get("augmentation_config", None)
        self.augmentation_seed = params.get("augmentation_seed", 42)

        # Initialize dataset attributes
        self.train_dataset = None
        self.train_labeled = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(
        self,
        stage: str | None = None,
        includeTestInTrain: bool = True,
        drop_column_list: list = [],
    ):
        """
        Setup datasets for training, validation, and testing.
        Args:
            stage (str | None): Stage of setup ("fit", "test", or None for all).
            includeTestInTrain (bool): Whether to include unlabeled test data in training set.
            drop_column_list (list): List of column names to drop before building the dataloader.
        """

        # Load and split datasets based on the stage
        # if stage is "fit", load training and validation datasets
        if stage == "fit" or stage is None:
            # Determine global features file path
            train_global_path = None
            if self.train_global_features_file:
                train_global_path = os.path.join(
                    self.data_dir, self.train_global_features_file
                )
            
            # Create augmentation pipeline if configured
            augmentation_pipeline = None
            if self.augmentation_config is not None:
                augmentation_pipeline = AugmentationPipeline.from_config(
                    self.augmentation_config, seed=self.augmentation_seed
                )

            # Load labeled training dataset WITHOUT windowing first
            # Windowing will be applied AFTER train/val split to prevent data leakage
            labeled_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.train_file_name),
                labelsPath=os.path.join(self.data_dir, self.train_file_name_labels),
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
                globalFeaturesPath=train_global_path,
                use_windowing=False,  # ← NO WINDOWING YET to prevent leakage
                window_size=self.window_size,
                stride=self.stride,
                drop_column_list=drop_column_list,
                augmentation_pipeline=augmentation_pipeline,
                is_training=True,  # Will be overridden for val dataset
            )

            # Update data info with non-windowed shape for now
            self.updateDataInfo(labeled_dataset)

            # Store full labeled dataset for K-Fold splitting (non-windowed)
            self._full_labeled_dataset = labeled_dataset

            # Split labeled dataset into train/val (do not mix unlabeled test into this split)
            # CRITICAL: Split happens on ORIGINAL SAMPLES, not windows
            # If using K-Fold, this will be overridden by setup_fold()
            val_size = int(len(labeled_dataset) * self.val_split)
            train_size = len(labeled_dataset) - val_size

            # Use random_split to get indices, then wrap with SubsetWithTrainingFlag
            train_subset_temp, val_subset_temp = random_split(
                labeled_dataset,
                [train_size, val_size],
                generator=self.trainValGenerator,
            )
            
            # Wrap subsets with proper is_training flags
            train_subset = SubsetWithTrainingFlag(labeled_dataset, train_subset_temp.indices, is_training=True)
            val_subset = SubsetWithTrainingFlag(labeled_dataset, val_subset_temp.indices, is_training=False)

            # NOW apply windowing separately to train and validation subsets
            # This prevents data leakage between sets
            if self.use_windowing:
                # Apply windowing to train subset (with augmentation)
                train_dataset_windowed = self._apply_windowing_to_subset(train_subset, is_training=True)
                self.train_labeled = train_dataset_windowed

                # Apply windowing to validation subset (NO augmentation)
                val_dataset_windowed = self._apply_windowing_to_subset(val_subset, is_training=False)
                self.val_dataset = val_dataset_windowed

                # Update data info with windowed shape
                self.updateDataInfo(train_dataset_windowed)
            else:
                # No windowing, use wrapped subsets directly
                self.train_labeled = train_subset
                self.val_dataset = val_subset

            if includeTestInTrain:
                # Determine test global features file path
                test_global_path = None
                if self.test_global_features_file:
                    test_global_path = os.path.join(
                        self.data_dir, self.test_global_features_file
                    )

                # Load unlabeled test dataset with same windowing setting as training
                # For autoencoder training, we can apply windowing AND augmentation to test data
                # Apply augmentation to test data too when training autoencoder (helps reconstruction)
                unlabeled_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                    dataPath=os.path.join(self.data_dir, self.test_file_name),
                    labelsPath=None,
                    labelMapping=list(self.label_mapping.keys()),
                    globalColumns=self.globalFeaturesColumns,
                    primaryKeyColumn=self.primaryKeyColumn,
                    timeSeriesColumns=self.timeSeriesColumns,
                    globalFeaturesPath=test_global_path,
                    use_windowing=self.use_windowing,  # Same windowing as train
                    window_size=self.window_size,
                    stride=self.stride,
                    drop_column_list=drop_column_list,
                    augmentation_pipeline=augmentation_pipeline,  # Apply augmentation for autoencoder
                    is_training=True,  # Enable augmentation for test data in autoencoder training
                )

                # For reconstruction training we allow unlabeled test data to be mixed with labeled train data
                self.train_dataset = ConcatDataset(
                    [self.train_labeled, unlabeled_dataset]
                )
            else:
                self.train_dataset = self.train_labeled

        if stage == "test" or stage is None:
            # Determine test global features file path
            test_global_path = None
            if self.test_global_features_file:
                test_global_path = os.path.join(
                    self.data_dir, self.test_global_features_file
                )

            # Load test dataset for inference - NO AUGMENTATION during prediction
            self.test_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.test_file_name),
                labelsPath=None,
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
                globalFeaturesPath=test_global_path,
                use_windowing=self.use_windowing,
                window_size=self.window_size,
                stride=self.stride,
                drop_column_list=drop_column_list,
                augmentation_pipeline=None,  # No augmentation for inference
                is_training=False,  # Disable augmentation during test/inference
            )

    def _apply_windowing_to_subset(self, subset: Subset, is_training: bool = True) -> TimeSeriesAndGlobalDataset:
        """
        Apply windowing to a subset of the dataset.

        This creates a new windowed dataset from the samples in the subset,
        preventing data leakage between train and validation sets.

        Args:
            subset: A Subset object containing indices into the original dataset
            is_training: Whether this is a training dataset (enables augmentation)

        Returns:
            A new TimeSeriesAndGlobalDataset with windowing applied
        """
        # Get the underlying dataset
        original_dataset = subset.dataset
        indices = subset.indices

        # Extract data for these specific indices
        if original_dataset.time_series_data is not None:
            subset_time_series = original_dataset.time_series_data[indices]
        else:
            subset_time_series = None

        subset_global = original_dataset.global_data[indices]

        if original_dataset.labels is not None:
            subset_labels = original_dataset.labels[indices]
        else:
            subset_labels = None

        # Create new dataset with windowing enabled
        # Augmentation only for training, not for validation/test
        windowed_dataset = TimeSeriesAndGlobalDataset(
            time_series_data=subset_time_series,
            global_data=subset_global,
            labels=subset_labels,
            use_windowing=True,  # Enable windowing
            window_size=self.window_size,
            stride=self.stride,
            augmentation_pipeline=original_dataset.augmentation_pipeline if is_training else None,
            is_training=is_training,
        )

        return windowed_dataset

    def updateDataInfo(self, dataset: TimeSeriesAndGlobalDataset) -> dict:
        # Update time series shape if windowing is enabled
        if dataset.use_windowing and dataset.timeSeriesShape is not None:
            # Time series shape becomes (features, window_size) instead of (features, original_seq_len)
            windowed_shape = (dataset.timeSeriesShape[0], dataset.window_size)
            timeSeriesShape = windowed_shape
        else:
            timeSeriesShape = dataset.timeSeriesShape

        self.dataInfo = {
            "timeSeriesShape": timeSeriesShape,
            "globalFeaturesShape": dataset.globalFeaturesShape,
            "numClasses": dataset.numClasses,
            "use_windowing": dataset.use_windowing,
            "window_size": dataset.window_size if dataset.use_windowing else None,
            "stride": dataset.stride if dataset.use_windowing else None,
        }
        return self.dataInfo

    def train_dataloader(self):
        """
        Create training data loader.
        """

        # Check if training dataset is initialized
        if self.train_dataset is None:
            raise RuntimeError("Training dataset not initialized. Call setup() first.")

        # Return the DataLoader for training dataset
        return TorchDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def val_dataloader(self):
        """
        Create validation data loader.
        """

        # Check if validation dataset is initialized
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )

        # Return the DataLoader for validation dataset
        return TorchDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def test_dataloader(self):
        """
        Create test data loader.
        """

        # Check if test dataset is initialized
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")

        # Return the DataLoader for test dataset
        return TorchDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            pin_memory=torch.cuda.is_available(),
        )

    def predict_dataloader(self):
        """
        Create prediction data loader (uses test dataset).
        """
        # Return the DataLoader for prediction dataset (which is the test dataset)
        return self.test_dataloader()

    def getDatasetInfo(self) -> dict:
        """
        Get information about the dataset, including input shapes and number of classes.
        Returns:
            dict: Dictionary containing dataset information.
        """

        return self.dataInfo

    def setup_fold(self, fold_idx: int, include_test_in_train: bool = True):
        """
        Setup train/val datasets for a specific K-Fold split.

        Args:
            fold_idx (int): Index of the fold to setup (0 to n_folds-1).
            include_test_in_train (bool): Whether to include unlabeled test data in training set.

        Note:
            Must call setup(stage='fit') before using this method to load the full dataset.
        """
        if not self.use_kfold:
            raise RuntimeError("K-Fold is not enabled. Set use_kfold=True in params.")

        if self._full_labeled_dataset is None:
            raise RuntimeError(
                "Full labeled dataset not loaded. Call setup(stage='fit') first."
            )

        if fold_idx < 0 or fold_idx >= self.n_folds:
            raise ValueError(
                f"fold_idx must be between 0 and {self.n_folds-1}, got {fold_idx}"
            )

        # Create K-Fold splitter
        kfold = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=42)

        # Get train/val indices for this fold
        all_indices = np.arange(len(self._full_labeled_dataset))
        splits = list(kfold.split(all_indices, self._full_labeled_dataset.labels.numpy()))
        train_indices, val_indices = splits[fold_idx]

        # Create train and val subsets with proper is_training flags
        # Use SubsetWithTrainingFlag to ensure different augmentation behavior
        train_subset = SubsetWithTrainingFlag(self._full_labeled_dataset, train_indices.tolist(), is_training=True)
        val_subset = SubsetWithTrainingFlag(self._full_labeled_dataset, val_indices.tolist(), is_training=False)

        # Apply windowing separately to prevent leakage
        if self.use_windowing:
            self.train_labeled = self._apply_windowing_to_subset(train_subset, is_training=True)
            self.val_dataset = self._apply_windowing_to_subset(val_subset, is_training=False)
        else:
            # For non-windowed datasets, just use the subsets directly
            self.train_labeled = train_subset
            self.val_dataset = val_subset

        # Load test dataset if needed and not already loaded
        if include_test_in_train and self.test_dataset is None:
            # Determine test global features file path
            test_global_path = None
            if self.test_global_features_file:
                test_global_path = os.path.join(
                    self.data_dir, self.test_global_features_file
                )

            # Load unlabeled test dataset with same windowing setting
            # Apply augmentation to test data for autoencoder reconstruction training
            # Use augmentation_pipeline if available (may be None if not configured)
            augmentation_pipeline = getattr(self, 'augmentation_pipeline', None)
            
            self.test_dataset = TimeSeriesAndGlobalDataset.fromCSV(
                dataPath=os.path.join(self.data_dir, self.test_file_name),
                labelsPath=None,
                labelMapping=list(self.label_mapping.keys()),
                globalColumns=self.globalFeaturesColumns,
                primaryKeyColumn=self.primaryKeyColumn,
                timeSeriesColumns=self.timeSeriesColumns,
                globalFeaturesPath=test_global_path,
                use_windowing=self.use_windowing,  # Same windowing as train
                window_size=self.window_size,
                stride=self.stride,
                augmentation_pipeline=augmentation_pipeline,  # Apply augmentation if configured
                is_training=True,  # Enable augmentation for autoencoder training
            )

        # Optionally include unlabeled test data in training
        if include_test_in_train and self.test_dataset is not None:
            self.train_dataset = ConcatDataset([self.train_labeled, self.test_dataset])
        else:
            self.train_dataset = self.train_labeled

        # Update current fold tracker
        self.current_fold = fold_idx

    def get_fold_info(self) -> dict:
        """
        Get information about the current K-Fold setup.

        Returns:
            dict: Dictionary containing K-Fold configuration and current fold index.
        """
        return {
            "use_kfold": self.use_kfold,
            "n_folds": self.n_folds,
            "current_fold": self.current_fold,
            "full_dataset_size": (
                len(self._full_labeled_dataset) if self._full_labeled_dataset else None
            ),
            "train_size": (
                len(self.train_labeled) if hasattr(self, "train_labeled") else None
            ),
            "val_size": len(self.val_dataset) if hasattr(self, "val_dataset") else None,
        }
