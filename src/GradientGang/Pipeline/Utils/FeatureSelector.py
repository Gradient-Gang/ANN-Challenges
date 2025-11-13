"""
Feature Selector for Global Features

Implements multiple feature selection methods:
- Variance Threshold
- Correlation-based filtering
- RandomForest feature importance
- Mutual Information
- L1 Regularization (Lasso)

Can select features using intersection of methods for robust selection.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import warnings

warnings.filterwarnings("ignore")


class FeatureSelector:
    """
    Feature selector that applies multiple selection methods and finds their intersection.

    Attributes:
        params (dict): Configuration parameters for feature selection
        selected_features_ (list): List of selected feature names after fitting
        feature_importances_ (dict): Dictionary of feature importance scores from different methods
    """

    def __init__(self, params: dict):
        """
        Initialize FeatureSelector with configuration parameters.

        Args:
            params (dict): Dictionary containing:
                - method: Selection method ('all_three_intersection', 'rf_top_k', 'mi_top_k', etc.)
                - variance_threshold: Threshold for variance-based filtering (default: 0.01)
                - correlation_threshold: Threshold for correlation-based filtering (default: 0.95)
                - top_k_rf: Number of top features to select by RandomForest (default: 300)
                - top_k_mi: Number of top features to select by Mutual Information (default: 300)
                - lasso_C: Regularization strength for Lasso (default: 0.1)
                - rf_n_estimators: Number of trees for RandomForest (default: 100)
                - rf_max_depth: Max depth for RandomForest (default: 10)
                - random_state: Random seed (default: 42)
        """
        self.params = params
        self.selected_features_ = None
        self.feature_importances_ = {}

        # Set default parameters
        self.variance_threshold = params.get("variance_threshold", 0.01)
        self.correlation_threshold = params.get("correlation_threshold", 0.95)
        self.top_k_rf = params.get("top_k_rf", 300)
        self.top_k_mi = params.get("top_k_mi", 300)
        self.lasso_C = params.get("lasso_C", 0.1)
        self.rf_n_estimators = params.get("rf_n_estimators", 100)
        self.rf_max_depth = params.get("rf_max_depth", 10)
        self.random_state = params.get("random_state", 42)
        self.method = params.get("method", "all_three_intersection")

    def _apply_variance_threshold(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove features with low variance."""
        selector = VarianceThreshold(threshold=self.variance_threshold)
        X_var = selector.fit_transform(X)
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"  Variance threshold: {X.shape[1]} → {len(selected_features)} features")
        return pd.DataFrame(X_var, columns=selected_features, index=X.index)

    def _apply_correlation_filtering(self, X: pd.DataFrame) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = X.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [
            col
            for col in upper_tri.columns
            if any(upper_tri[col] > self.correlation_threshold)
        ]
        X_filtered = X.drop(columns=to_drop)
        print(f"  Correlation filtering: {X.shape[1]} → {X_filtered.shape[1]} features")
        return X_filtered

    def _select_rf_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Select top features using RandomForest importance."""
        print(f"  Training RandomForest for feature selection...")
        rf = RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            random_state=self.random_state,
            n_jobs=-1,
        )
        rf.fit(X, y)

        importances = pd.DataFrame(
            {"feature": X.columns, "importance": rf.feature_importances_}
        ).sort_values("importance", ascending=False)

        self.feature_importances_["rf"] = importances
        selected = importances.head(self.top_k_rf)["feature"].tolist()
        print(f"  RandomForest selected top {len(selected)} features")
        return selected

    def _select_mi_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Select top features using Mutual Information."""
        print(f"  Calculating Mutual Information scores...")
        mi_scores = mutual_info_classif(
            X, y, random_state=self.random_state, n_neighbors=3
        )

        mi_df = pd.DataFrame({"feature": X.columns, "mi_score": mi_scores}).sort_values(
            "mi_score", ascending=False
        )

        self.feature_importances_["mi"] = mi_df
        selected = mi_df.head(self.top_k_mi)["feature"].tolist()
        print(f"  Mutual Information selected top {len(selected)} features")
        return selected

    def _select_lasso_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Select features using L1 regularization (Lasso)."""
        print(f"  Training Lasso for feature selection...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        lasso = LogisticRegression(
            penalty="l1",
            C=self.lasso_C,
            solver="saga",
            max_iter=500,
            random_state=self.random_state,
            n_jobs=-1,
        )
        lasso.fit(X_scaled, y)

        coefficients = np.abs(lasso.coef_).sum(axis=0)
        lasso_df = pd.DataFrame(
            {"feature": X.columns, "coefficient": coefficients}
        ).sort_values("coefficient", ascending=False)

        self.feature_importances_["lasso"] = lasso_df
        selected = lasso_df[lasso_df["coefficient"] > 0]["feature"].tolist()
        print(f"  Lasso selected {len(selected)} non-zero features")
        return selected

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the feature selector on training data.

        Args:
            X (pd.DataFrame): Feature matrix (samples × features)
            y (pd.Series): Target labels

        Returns:
            self: Fitted FeatureSelector instance
        """
        print(f"\n{'='*60}")
        print(f"Feature Selection: {self.method}")
        print(f"{'='*60}")
        print(f"Input features: {X.shape[1]}")

        # Step 1: Variance threshold
        X_var = self._apply_variance_threshold(X)

        # Step 2: Correlation filtering
        X_corr = self._apply_correlation_filtering(X_var)

        # Step 3: Apply selection method
        if self.method == "variance_correlation_only":
            self.selected_features_ = X_corr.columns.tolist()

        elif self.method == "rf_top_k":
            rf_features = self._select_rf_features(X_corr, y)
            self.selected_features_ = rf_features

        elif self.method == "mi_top_k":
            mi_features = self._select_mi_features(X_corr, y)
            self.selected_features_ = mi_features

        elif self.method == "lasso_nonzero":
            lasso_features = self._select_lasso_features(X_corr, y)
            self.selected_features_ = lasso_features

        elif self.method == "rf_union_mi":
            rf_features = self._select_rf_features(X_corr, y)
            mi_features = self._select_mi_features(X_corr, y)
            self.selected_features_ = list(set(rf_features) | set(mi_features))

        elif self.method == "all_three_intersection":
            rf_features = self._select_rf_features(X_corr, y)
            mi_features = self._select_mi_features(X_corr, y)
            lasso_features = self._select_lasso_features(X_corr, y)

            rf_set = set(rf_features)
            mi_set = set(mi_features)
            lasso_set = set(lasso_features)

            self.selected_features_ = list(rf_set & mi_set & lasso_set)

        elif self.method == "at_least_two":
            rf_features = self._select_rf_features(X_corr, y)
            mi_features = self._select_mi_features(X_corr, y)
            lasso_features = self._select_lasso_features(X_corr, y)

            rf_set = set(rf_features)
            mi_set = set(mi_features)
            lasso_set = set(lasso_features)

            at_least_two = (
                (rf_set & mi_set) | (rf_set & lasso_set) | (mi_set & lasso_set)
            )
            self.selected_features_ = list(at_least_two)

        else:
            raise ValueError(f"Unknown selection method: {self.method}")

        print(f"\n{'='*60}")
        print(f"✓ Feature selection completed")
        print(f"  Final selected features: {len(self.selected_features_)}")
        print(
            f"  Reduction: {X.shape[1]} → {len(self.selected_features_)} ({100*(1-len(self.selected_features_)/X.shape[1]):.1f}% reduction)"
        )
        print(f"{'='*60}\n")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data by selecting only the fitted features.

        Args:
            X (pd.DataFrame): Feature matrix to transform

        Returns:
            pd.DataFrame: Transformed feature matrix with selected features only
        """
        if self.selected_features_ is None:
            raise ValueError(
                "FeatureSelector must be fitted before transform. Call fit() first."
            )

        # Check if all selected features are present
        missing_features = set(self.selected_features_) - set(X.columns)
        if missing_features:
            raise ValueError(f"Missing features in data: {missing_features}")

        return X[self.selected_features_]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Fit the selector and transform data in one step.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target labels

        Returns:
            pd.DataFrame: Transformed feature matrix
        """
        self.fit(X, y)
        return self.transform(X)

    def save_selected_features(self, filepath: str):
        """
        Save selected feature names to a text file.

        Args:
            filepath (str): Path to save the feature list
        """
        if self.selected_features_ is None:
            raise ValueError("No features selected. Call fit() first.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write("\n".join(self.selected_features_))

        print(f"✓ Saved {len(self.selected_features_)} selected features to {filepath}")

    def load_selected_features(self, filepath: str):
        """
        Load selected feature names from a text file.

        Args:
            filepath (str): Path to load the feature list from
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature list file not found: {filepath}")

        with open(filepath, "r") as f:
            self.selected_features_ = [line.strip() for line in f if line.strip()]

        print(
            f"✓ Loaded {len(self.selected_features_)} selected features from {filepath}"
        )

    def get_feature_importance_df(self, method: str = "rf") -> pd.DataFrame:
        """
        Get feature importance scores from a specific method.

        Args:
            method (str): Method name ('rf', 'mi', or 'lasso')

        Returns:
            pd.DataFrame: Feature importance scores
        """
        if method not in self.feature_importances_:
            raise ValueError(
                f"Method '{method}' not available. Available methods: {list(self.feature_importances_.keys())}"
            )

        return self.feature_importances_[method]
