import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
from sklearn.decomposition import PCA
from tqdm.notebook import tqdm
import seaborn as sns
import sys
from imblearn.over_sampling import SMOTE, RandomOverSampler

from ..Pipeline.Utils.FeatureSelector import FeatureSelector

sns.set_theme()


class PreProcessor:
    @staticmethod
    def fromYAML(path: str):
        """
        Builds a PreProcessor from a YAML file.
        Args:
            path (str): The path of the YAML file
        Returns:
            PreProcessor: The built PreProcessor
        """

        # Load parameters from YAML file
        with open(path, "r") as f:
            params = yaml.safe_load(f)

        # Create and return PreProcessor instance
        return PreProcessor(params)

    def __init__(self, params: dict):
        """
        Initialize the PreProcessor with given parameters.
        Args:
            params (dict): Dictionary of parameters for preprocessing
        """

        # Store parameters
        self.params = params

        # paths
        self.path_raw_data = self.params.get("path_raw_data", "")
        self.path_processed_data = self.params.get("path_processed_data", "")

        # file names
        self.name_train_file = self.params.get("name_train_file", "train.csv")
        self.name_test_file = self.params.get("name_test_file", "test.csv")
        self.name_train_labels_file = self.params.get(
            "name_train_labels_file", "train_labels.csv"
        )

        # feature handling flags
        self.drop_all_is_pirate = self.params.get("drop_all_is_pirate", False)
        self.one_hot_encode_is_pirate = self.params.get("one_hot_encode", False)

        # PCA
        self.use_pca = self.params.get("PCA", False)
        if self.use_pca:
            self.explained_variance = self.params.get("explained_variance", 0.95)

        # feature selection
        self.use_feature_selection = self.params.get("feature_selection", False)
        if self.use_feature_selection:
            self.feature_selected = self.params.get("feature_selected", None)

        # oversampling
        self.use_oversampling = self.params.get("use_oversampling", False)
        if self.use_oversampling:
            self.oversampling_method = self.params.get("oversampling_method", "SMOTE")
            self.oversampling_random_state = self.params.get("oversampling_random_state", 42)

        # Columns not to be normalized
        self.columns_excluded_from_normalization: list[str] = self.params.get(
            "columns_excluded_from_normalization",
            ["sample_index", "time", "isPirate", "isNotPirate"],
        )

        # verbosity
        self.verbose = self.params.get("verbose", True)

    def load_data(self, file_name: str) -> pd.DataFrame:
        """
        Load data from a CSV file.
        Args:
            file_name (str): The name of the CSV file to load
        Returns:
            pd.DataFrame: The loaded data as a DataFrame
        """

        # Construct full file path
        file_path = os.path.join(self.path_raw_data, file_name)

        # Load data from CSV
        data = pd.read_csv(file_path)
        return data

    def save_data(self, data, file_name: str):
        """
        Save data to a CSV file. Accepts DataFrame or NumPy array.
        Args:
            data (pd.DataFrame or np.ndarray): The data to save
            file_name (str): The name of the CSV file to save to
        """
        os.makedirs(self.path_processed_data, exist_ok=True)

        # Construct full file path
        file_path = os.path.join(self.path_processed_data, file_name)
        # Convert to DataFrame if necessary
        if not isinstance(data, pd.DataFrame):
            # convert NumPy array or other array-like to DataFrame
            data = pd.DataFrame(data)

        # Save data to CSV
        data.to_csv(file_path, index=False)

    def remove_last_column(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove the last column from the DataFrame.
        Args:
            data (pd.DataFrame): The input DataFrame
        Returns:
            pd.DataFrame: DataFrame with the last column removed
        """

        # Remove the last column from the DataFrame
        return data.drop(data.columns[-1], axis=1)

    def handle_is_pirate_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle isPirate features by dropping or encoding them.
        Args:
            data (pd.DataFrame): The input DataFrame
        Returns:
            pd.DataFrame: DataFrame with isPirate features handled
        """

        # If user wants to drop everything related to 'is_pirate' features
        if self.drop_all_is_pirate:
            return data.drop(columns=["n_legs", "n_hands", "n_eyes"], errors="ignore")

        # default: drop n_legs and n_hands
        data = data.drop(columns=["n_legs", "n_hands"], errors="ignore")

        # default: map n_eyes to a numeric column called 'number' then drop originals
        # Map eye categories to numeric values
        # 0: two eyes, 1: one eye or eye patch, 2: no eyes (default)
        eye_map = {"two": 0, "one+eye_patch": 1}
        data["isPirate"] = data["n_eyes"].map(eye_map).fillna(0).astype(int)
        data = data.drop(columns=["n_eyes"], errors="ignore")

        # One-hot encode if specified
        if self.one_hot_encode_is_pirate:
            data["isNotPirate"] = 1 - data["isPirate"]

        # Return the modified DataFrame
        return data

    def normalize_per_process(
        self, training_data: pd.DataFrame, test_data: pd.DataFrame
    ):
        """
        Normalize numerical features to have zero mean and unit variance.
        Args:
            training_data (pd.DataFrame): The training data
            test_data (pd.DataFrame): The test data
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Normalized training and test data
        """
        # Select numeric columns (includes integers and floats)
        columns = training_data.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude specified columns from normalization
        for col in self.columns_excluded_from_normalization:
            if col in columns:
                columns.remove(col)

        # Normalize each column with safe handling for zero std
        for col in columns:
            mean = training_data[col].mean()
            std = training_data[col].std()
            if std == 0 or np.isnan(std):
                # center only; keep zeros for constant feature in train, center test
                training_data[col] = training_data[col] - mean
                test_data[col] = test_data[col] - mean
            else:
                training_data[col] = (training_data[col] - mean) / std
                test_data[col] = (test_data[col] - mean) / std

        return training_data, test_data

    def extract_global_features(
        self,
        data: pd.DataFrame,
        feature_types: list[str] = ["statistical", "trend", "domain"],
    ) -> pd.DataFrame:
        """
        Extract global statistical and trend features from time series data.

        Args:
            data (pd.DataFrame): Time series data with sample_index, time, and feature columns
            feature_types (list[str]): Types of features to extract

        Returns:
            pd.DataFrame: Global features with one row per sample_index
        """
        # Identify time series columns (exclude sample_index, time, isPirate, isNotPirate)
        excluded_cols = ["sample_index", "time", "isPirate", "isNotPirate"]
        time_series_cols = [col for col in data.columns if col not in excluded_cols]

        # Get unique sample indices
        sample_indices = data["sample_index"].unique()

        # List to store feature dictionaries for each sample
        features_list = []

        for sample_id in sample_indices:
            sample_data = data[data["sample_index"] == sample_id].copy()
            feature_dict = {"sample_index": sample_id}

            # Extract features for each time series column
            for col in time_series_cols:
                values = sample_data[col].values
                time_values = sample_data["time"].values

                # Statistical features
                if "statistical" in feature_types:
                    feature_dict[f"{col}_mean"] = np.mean(values)
                    feature_dict[f"{col}_std"] = np.std(values)
                    feature_dict[f"{col}_min"] = np.min(values)
                    feature_dict[f"{col}_max"] = np.max(values)
                    feature_dict[f"{col}_range"] = np.max(values) - np.min(values)

                # Trend features
                if "trend" in feature_types:
                    try:
                        # Linear regression slope
                        slope, _ = np.polyfit(time_values, values, 1)
                        feature_dict[f"{col}_slope"] = slope
                    except:
                        feature_dict[f"{col}_slope"] = 0.0

                    feature_dict[f"{col}_first"] = values[0] if len(values) > 0 else 0.0
                    feature_dict[f"{col}_last"] = values[-1] if len(values) > 0 else 0.0
                    feature_dict[f"{col}_change"] = (
                        values[-1] - values[0] if len(values) > 0 else 0.0
                    )

            # Domain-specific features
            if "domain" in feature_types:
                # Joint activity features
                joint_cols = [col for col in time_series_cols if "joint" in col]
                if joint_cols:
                    joint_means = [
                        feature_dict[f"{col}_mean"]
                        for col in joint_cols
                        if f"{col}_mean" in feature_dict
                    ]
                    feature_dict["joint_activity_mean"] = (
                        np.mean(joint_means) if joint_means else 0.0
                    )

            features_list.append(feature_dict)

        # Convert to DataFrame
        global_features_df = pd.DataFrame(features_list)

        return global_features_df

    def plot_one_time_series(self, data: pd.DataFrame, number: int):
        """
        Plot a single time series from the DataFrame.
        Args:
            data (pd.DataFrame): The input DataFrame
            number (int): Number of time series to plot
        """

        # Select sample indices to plot
        Set = [i for i in range(number)]
        T = [data[data["sample_index"] == i] for i in Set]

        print(f"Selected sample indices: {Set}")

        # Create subplots for different features (only joint features now)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        axes = axes.flatten()

        # Plot each time series on the respective subplot
        for ti in T:
            axes[0].plot(ti["time"], ti["joint_00"])
            axes[1].plot(ti["time"], ti["joint_01"])
            axes[2].plot(ti["time"], ti["joint_28"])
            axes[3].plot(ti["time"], ti["joint_29"])

        axes[0].set(title="joint 00", xlabel="time", ylabel="joint_00")
        axes[1].set(title="joint 01", xlabel="time", ylabel="joint_01")
        axes[2].set(title="joint 28", xlabel="time", ylabel="joint_28")
        axes[3].set(title="joint 29", xlabel="time", ylabel="joint_29")

        plt.tight_layout()
        plt.show()

    def __aggregate_time_series(
        self,
        data: pd.DataFrame,
        primaryKeyColumn: str,
        timeColumn: str,
        extraColumns: list[str],
    ) -> np.ndarray:
        """
        Aggregate time series data into a 3D NumPy array.
        Args:
            data (pd.DataFrame): The input DataFrame
            primaryKeyColumn (str): The primary key column name
            timeColumn (str): The time column name
            extraColumns (list[str]): Columns to exclude from aggregation
        Returns:
            np.ndarray: 3D NumPy array of shape (samples, time_steps, features)
        """

        # Remove extra columns
        data = data[data.columns.difference(extraColumns)]

        # Pivot and stack data into 3D NumPy array
        npData = np.stack(
            [
                data.pivot(index="sample_index", columns="time", values=feat).to_numpy()
                for feat in data.columns.difference([primaryKeyColumn, timeColumn])
            ],
            axis=-1,
        )

        # Return the aggregated NumPy array
        return npData

    def __plotPcaNumComponentsPerFeature(
        self, pcadata: list[np.ndarray], feature_names: list[str]
    ):
        """
        Plot the number of PCA components selected per feature.
        Args:
            pcadata (list[np.ndarray]): List of PCA-transformed data arrays
            feature_names (list[str]): List of feature names
        """
        num_components = [data.shape[1] for data in pcadata]

        plt.figure(figsize=(10, 6))
        plt.bar(
            feature_names,
            num_components,
            color=sns.color_palette("viridis", len(feature_names)),
            align="edge",
        )
        plt.xlabel("Feature")
        plt.ylabel("Number of PCA Components")
        plt.title("PCA Components per Feature")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def apply_pca(self, training_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Apply PCA (Proper Orthogonal Decomposition) to time series to extract global features.
        This method extracts POD features from time series and returns them as additional global features.

        Args:
            training_data (pd.DataFrame): The training data
            test_data (pd.DataFrame): The test data

        Returns:
            tuple: Tuple containing DataFrames with POD global features (one row per sample)
        """

        # Aggregate time series data into 3D NumPy arrays
        trainingDataNp = self.__aggregate_time_series(
            training_data, "sample_index", "time", ["isPirate", "isNotPirate"]
        )

        # Aggregate test data into 3D NumPy array
        testDataNp = self.__aggregate_time_series(
            test_data, "sample_index", "time", ["isPirate", "isNotPirate"]
        )

        # Get the number of training samples
        maxTrainingIndex = trainingDataNp.shape[0]

        # Concatenate training and test data for PCA
        dataNp = np.concatenate((trainingDataNp, testDataNp), axis=0)

        pcaData = []

        for i in tqdm(range(dataNp.shape[2]), desc="Applying PCA"):
            # Apply PCA on each feature separately
            featureSlice = dataNp[:, :, i]

            pca = PCA(n_components=self.explained_variance)
            featureSliceTransformed = pca.fit_transform(featureSlice)

            pcaData.append(featureSliceTransformed)

        # Verbose mode: plot the number of PCA components per feature
        if self.verbose:
            feature_names = training_data.columns.difference(
                ["sample_index", "time", "isPirate", "isNotPirate"]
            ).tolist()
            self.__plotPcaNumComponentsPerFeature(pcaData, feature_names)

        # Concatenate PCA-transformed data for all features
        pcaData = np.concatenate(pcaData, axis=1)

        trainingPcaData = pcaData[:maxTrainingIndex]
        testPcaData = pcaData[maxTrainingIndex:]

        # Create DataFrames with POD features
        # Add POD_ prefix to distinguish from other global features
        num_components = trainingPcaData.shape[1]
        pod_columns = [f"POD_{i}" for i in range(num_components)]

        training_pod_df = pd.DataFrame(trainingPcaData, columns=pod_columns)
        training_pod_df.insert(0, "sample_index", range(len(training_pod_df)))

        test_pod_df = pd.DataFrame(testPcaData, columns=pod_columns)
        test_pod_df.insert(0, "sample_index", range(len(test_pod_df)))

        # Return POD global features (one row per sample)
        return training_pod_df, test_pod_df

    def apply_feature_selection(
        self,
        train_global_features: pd.DataFrame,
        test_global_features: pd.DataFrame,
        train_labels: pd.Series,
    ):
        """
        Apply feature selection to global features.

        Args:
            train_global_features (pd.DataFrame): Training global features (with sample_index)
            test_global_features (pd.DataFrame): Test global features (with sample_index)
            train_labels (pd.Series): Training labels for supervised feature selection

        Returns:
            tuple: (train_selected, test_selected) with selected features
        """
        # Get feature selection parameters
        fs_params = self.params.get("feature_selection_params", {})

        if not fs_params:
            print(
                "Warning: No feature_selection_params found in config. Using defaults."
            )
            fs_params = {
                "method": "all_three_intersection",
                "variance_threshold": 0.01,
                "correlation_threshold": 0.95,
                "top_k_rf": 300,
                "top_k_mi": 300,
                "random_state": 42,
            }

        # Initialize FeatureSelector
        selector = FeatureSelector(fs_params)

        # Separate sample_index from features (must be preserved)
        train_X = train_global_features.drop(columns=["sample_index"])
        test_X = test_global_features.drop(columns=["sample_index"])

        # Fit on training data and transform both
        train_X_selected = selector.fit_transform(train_X, train_labels)
        test_X_selected = selector.transform(test_X)

        # Reconstruct with sample_index preserved
        train_selected = pd.concat(
            [
                train_global_features[["sample_index"]].reset_index(drop=True),
                train_X_selected.reset_index(drop=True),
            ],
            axis=1,
        )

        # Reconstruct test data with sample_index and selected features
        test_selected = pd.concat(
            [
                test_global_features[["sample_index"]].reset_index(drop=True),
                test_X_selected.reset_index(drop=True),
            ],
            axis=1,
        )

        # Save selected feature names if path specified
        save_path = fs_params.get("selected_features_file", None)
        if save_path:
            full_path = os.path.join(self.path_processed_data, save_path)
            selector.save_selected_features(full_path)

        return train_selected, test_selected

    def apply_oversampling(self, train_data: pd.DataFrame, train_labels: pd.DataFrame):
        """
        Apply oversampling to handle class imbalance in time series data.
        
        Args:
            train_data (pd.DataFrame): Training time series data with sample_index, time columns
            train_labels (pd.DataFrame): Training labels
            
        Returns:
            tuple: (oversampled_train_data, oversampled_train_labels)
        """
        print(f"\nApplying {self.oversampling_method} oversampling...")
        
        # Map labels to numeric
        label_map = {"no_pain": 0, "low_pain": 1, "high_pain": 2}
        train_labels_numeric = train_labels["label"].map(label_map).values
        
        # Get unique sample indices
        sample_indices = train_data["sample_index"].unique()
        
        # Create feature matrix: one row per sample with all timesteps flattened
        # Exclude sample_index and time from features
        feature_cols = [col for col in train_data.columns if col not in ["sample_index", "time"]]
        
        # Pivot to get shape (n_samples, n_timesteps * n_features)
        X_list = []
        for idx in sample_indices:
            sample_data = train_data[train_data["sample_index"] == idx][feature_cols].values.flatten()
            X_list.append(sample_data)
        
        X = np.array(X_list)
        y = train_labels_numeric
        
        print(f"Original data shape: {X.shape}, Label distribution: {np.bincount(y)}")
        
        # Apply oversampling
        if self.oversampling_method == "SMOTE":
            oversampler = SMOTE(random_state=self.oversampling_random_state)
        elif self.oversampling_method == "RandomOverSampler":
            oversampler = RandomOverSampler(random_state=self.oversampling_random_state)
        else:
            raise ValueError(f"Unknown oversampling method: {self.oversampling_method}")
        
        X_resampled, y_resampled = oversampler.fit_resample(X, y)
        
        print(f"Resampled data shape: {X_resampled.shape}, Label distribution: {np.bincount(y_resampled)}")
        
        # Reconstruct time series DataFrame
        n_timesteps = train_data["time"].nunique()
        n_features = len(feature_cols)
        
        # Create new DataFrame with oversampled data
        new_data_list = []
        for new_idx, sample_flat in enumerate(X_resampled):
            # Reshape back to (n_timesteps, n_features)
            sample_reshaped = sample_flat.reshape(n_timesteps, n_features)
            
            # Create DataFrame for this sample
            sample_df = pd.DataFrame(sample_reshaped, columns=feature_cols)
            sample_df["sample_index"] = new_idx
            sample_df["time"] = range(n_timesteps)
            
            new_data_list.append(sample_df)
        
        train_data_resampled = pd.concat(new_data_list, ignore_index=True)
        
        # Reconstruct labels DataFrame
        reverse_label_map = {0: "no_pain", 1: "low_pain", 2: "high_pain"}
        train_labels_resampled = pd.DataFrame({
            "sample_index": range(len(y_resampled)),
            "label": [reverse_label_map[label] for label in y_resampled]
        })
        
        return train_data_resampled, train_labels_resampled

    def preprocess(self):
        """
        Main preprocessing function to load, process, and save data.
        """

        # Load data
        try:
            train_data = self.load_data(self.name_train_file)
            test_data = self.load_data(self.name_test_file)
            train_labels = self.load_data(self.name_train_labels_file)
        except Exception as e:
            print(f"Error loading data: {e}")
            return
        print("Data loaded successfully.")

        # Remove pain survey features and isPirate-related columns
        cols_to_remove = ['pain_survey_1', 'pain_survey_2', 'pain_survey_3', 'pain_survey_4', 'n_legs', 'n_hands', 'n_eyes']
        train_data = train_data.drop(columns=[col for col in cols_to_remove if col in train_data.columns], errors='ignore')
        test_data = test_data.drop(columns=[col for col in cols_to_remove if col in test_data.columns], errors='ignore')
        print(f"Removed pain survey and isPirate features: {cols_to_remove}")

        # Remove last column
        try:
            train_data = self.remove_last_column(train_data)
            test_data = self.remove_last_column(test_data)
        except Exception as e:
            print(f"Error removing last column: {e}")
            return
        print("Last column removed successfully.")

        # isPirate features already removed - skip handling
        print("isPirate features already removed.")

        # Plot one time series if verbose
        if self.verbose:
            try:
                self.plot_one_time_series(train_data, number=3)
            except Exception:
                # plotting should not stop preprocessing
                pass

        # Normalize data
        try:
            train_data, test_data = self.normalize_per_process(train_data, test_data)
        except Exception as e:
            print(f"Error normalizing data: {e}")
            return
        print("Data normalized successfully.")

        # Plot one time series if verbose after normalization
        if self.verbose:
            try:
                self.plot_one_time_series(train_data, number=3)
            except Exception:
                pass

        # Extract global features if specified
        extract_global = self.params.get("extract_global_features", False)
        if extract_global:
            try:
                feature_types = self.params.get(
                    "global_features_to_extract", ["statistical", "trend", "domain"]
                )

                # Extract from training data
                train_global_features = self.extract_global_features(
                    train_data, feature_types
                )

                # Extract from test data
                test_global_features = self.extract_global_features(
                    test_data, feature_types
                )

                print(f"Global features extracted successfully.")
                print(
                    f"  - Training: {train_global_features.shape[1] - 1} features from {train_global_features.shape[0]} samples"
                )
                print(
                    f"  - Test: {test_global_features.shape[1] - 1} features from {test_global_features.shape[0]} samples"
                )

            except Exception as e:
                print(f"Error extracting global features: {e}")
                # Set to None to skip merging later
                train_global_features = None
                test_global_features = None
        else:
            print("Global feature extraction not enabled.")
            train_global_features = None
            test_global_features = None

        # Apply PCA if specified (Proper Orthogonal Decomposition on time series)
        if self.use_pca:
            try:
                train_pod_features, test_pod_features = self.apply_pca(
                    train_data, test_data
                )
                print(f"POD (PCA) applied successfully.")
                print(f"  - POD components: {train_pod_features.shape[1] - 1}")
            except Exception as e:
                print(f"Error applying PCA: {e}")
                train_pod_features = None
                test_pod_features = None
        else:
            print("PCA not applied.")
            train_pod_features = None
            test_pod_features = None

        # Merge all global features (statistical/trend + POD)
        # Create base DataFrames with just sample_index
        unique_train_samples = train_data["sample_index"].unique()
        unique_test_samples = test_data["sample_index"].unique()
        
        train_global_combined = pd.DataFrame({"sample_index": unique_train_samples})
        test_global_combined = pd.DataFrame({"sample_index": unique_test_samples})

        # Merge statistical/trend global features if available
        if train_global_features is not None:
            train_global_combined = train_global_combined.merge(
                train_global_features, on="sample_index", how="left"
            )
            test_global_combined = test_global_combined.merge(
                test_global_features, on="sample_index", how="left"
            )

        # Merge POD features if available
        if train_pod_features is not None:
            train_global_combined = train_global_combined.merge(
                train_pod_features, on="sample_index", how="left"
            )
            test_global_combined = test_global_combined.merge(
                test_pod_features, on="sample_index", how="left"
            )

        # Apply feature selection on global features if specified
        if self.use_feature_selection:
            try:
                # Map labels to numeric for supervised feature selection
                label_map = {"no_pain": 0, "low_pain": 1, "high_pain": 2}
                train_labels_numeric = train_labels["label"].map(label_map)

                print("\nApplying feature selection to global features...")
                train_global_combined, test_global_combined = (
                    self.apply_feature_selection(
                        train_global_combined,
                        test_global_combined,
                        train_labels_numeric,
                    )
                )
            except Exception as e:
                print(f"Error applying feature selection: {e}")
                import traceback

                traceback.print_exc()
                # Continue with unselected features
        else:
            print("Feature selection not applied.")

        # Save final global features (selected or full)
        try:
            self.save_data(train_global_combined, "train_global_features.csv")
            self.save_data(test_global_combined, "test_global_features.csv")
            print(f"Global features saved successfully.")
            print(f"  - Total global features: {train_global_combined.shape[1] - 1}")
        except Exception as e:
            print(f"Error saving global features: {e}")

        # Apply oversampling if specified (before saving time series data)
        # Note: Global features are NOT oversampled to maintain consistency with feature selection
        # Oversampling is applied only to time series data
        if self.use_oversampling:
            try:
                print("\n⚠️  WARNING: Oversampling time series data only.")
                print("   Global features will NOT be oversampled to maintain feature selection consistency.")
                print("   The DataLoader will need to handle the mismatch in sample counts.\n")
                
                train_data, train_labels = self.apply_oversampling(train_data, train_labels)
                
            except Exception as e:
                print(f"Error applying oversampling: {e}")
                import traceback
                traceback.print_exc()
        
        self.save_data(train_data, self.name_train_file)
        self.save_data(test_data, self.name_test_file)
        self.save_data(train_labels, self.name_train_labels_file)

        self.computeAndSaveClassWeights(
            train_labels,
            savingPath=os.path.join(self.path_processed_data, "class_weights.yaml"),
        )
        print("Data saved successfully.")

    def computeAndSaveClassWeights(
        self, labels: pd.DataFrame, savingPath: str = "class_weights.yaml"
    ):
        """
        Compute and save class weights to handle class imbalance.
        Args:
            labels (pd.DataFrame): DataFrame containing the labels
        """
        labels_mapping = {"no_pain": 0, "low_pain": 1, "high_pain": 2}
        class_counts = labels["label"].value_counts().to_dict()
        total_samples = len(labels)
        class_weights = {
            cls: total_samples / (len(class_counts) * count)
            for cls, count in class_counts.items()
        }
        class_weights = {
            labels_mapping[cls]: weight for cls, weight in class_weights.items()
        }

        # save on a file
        with open(savingPath, "w") as f:
            yaml.dump(class_weights, f)
