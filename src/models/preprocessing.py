"""
Feature engineering and preprocessing utilities for revenue prediction model.
"""
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any


class FeatureEngineer:
    """
    Handles feature engineering for the revenue prediction model.
    Includes both training and inference transformations.
    """

    def __init__(self, metadata: Dict[str, Any] = None):
        """
        Initialize FeatureEngineer.

        Args:
            metadata: Dictionary containing encoding mappings and statistics
        """
        self.metadata = metadata or {}
        self.feature_cols = self.metadata.get('feature_cols', [])

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create derived features from raw events.

        Args:
            df: DataFrame with raw features

        Returns:
            DataFrame with additional derived features
        """
        df = df.copy()

        # Handle missing values in event_3
        if 'event_3' in df.columns:
            df['event_3'].fillna(0, inplace=True)

        # Create total events
        df['total_events'] = df['event_1'] + df['event_2'] + df['event_3']

        # Create event ratios
        df['event_1_ratio'] = df['event_1'] / (df['total_events'] + 1)
        df['event_2_ratio'] = df['event_2'] / (df['total_events'] + 1)
        df['event_3_ratio'] = df['event_3'] / (df['total_events'] + 1)

        return df

    def encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using frequency and target encoding.

        Args:
            df: DataFrame with categorical features

        Returns:
            DataFrame with encoded features
        """
        df = df.copy()

        # Normalize platform
        if 'platform' in df.columns:
            df['platform'] = df['platform'].str.lower()

        # Frequency encoding using pre-computed counts
        if 'country' in df.columns and 'country_value_counts' in self.metadata:
            country_counts = self.metadata['country_value_counts']
            df['country_freq'] = df['country'].map(country_counts).fillna(1)

        if 'device_family' in df.columns and 'device_family_value_counts' in self.metadata:
            device_counts = self.metadata['device_family_value_counts']
            df['device_family_freq'] = df['device_family'].map(device_counts).fillna(1)

        if 'country_region' in df.columns and 'country_region_value_counts' in self.metadata:
            region_counts = self.metadata['country_region_value_counts']
            df['country_region_freq'] = df['country_region'].map(region_counts).fillna(1)

        # Target encoding for country
        if 'country' in df.columns and 'country_mean_revenue' in self.metadata:
            country_mean_rev = self.metadata['country_mean_revenue']
            global_mean = np.mean(list(country_mean_rev.values()))
            df['country_mean_revenue'] = df['country'].map(country_mean_rev).fillna(global_mean)

        return df

    def transform(self, df: pd.DataFrame,
                  le_source=None, le_platform=None) -> pd.DataFrame:
        """
        Apply full feature engineering pipeline.

        Args:
            df: Raw input DataFrame
            le_source: LabelEncoder for source (required for inference)
            le_platform: LabelEncoder for platform (required for inference)

        Returns:
            DataFrame with engineered features ready for model
        """
        # Create derived features
        df = self.create_derived_features(df)

        # Encode categorical features
        df = self.encode_categorical(df)

        # Label encoding for low cardinality features
        if le_source is not None and 'source' in df.columns:
            # Handle unseen categories
            df['source_encoded'] = df['source'].map(
                lambda x: le_source.transform([x])[0]
                if x in le_source.classes_ else -1
            )

        if le_platform is not None and 'platform' in df.columns:
            # Handle unseen categories
            df['platform_encoded'] = df['platform'].map(
                lambda x: le_platform.transform([x])[0]
                if x in le_platform.classes_ else -1
            )

        # Select only required features
        if self.feature_cols:
            # Ensure all required columns exist
            for col in self.feature_cols:
                if col not in df.columns:
                    df[col] = 0  # Default value for missing features

            return df[self.feature_cols]

        return df


def load_model_artifacts(model_dir: str = 'src/models/artifacts'):
    """
    Load model and all associated artifacts.

    Args:
        model_dir: Directory containing model artifacts

    Returns:
        Tuple of (model, feature_engineer, le_source, le_platform)
    """
    model = joblib.load(f'{model_dir}/model.pkl')
    metadata = joblib.load(f'{model_dir}/metadata.pkl')
    le_source = joblib.load(f'{model_dir}/le_source.pkl')
    le_platform = joblib.load(f'{model_dir}/le_platform.pkl')

    feature_engineer = FeatureEngineer(metadata)

    return model, feature_engineer, le_source, le_platform


def predict_revenue(input_data: Dict[str, Any],
                   model, feature_engineer, le_source, le_platform) -> float:
    """
    Make a single revenue prediction.

    Args:
        input_data: Dictionary with user features
        model: Trained model
        feature_engineer: FeatureEngineer instance
        le_source: Source label encoder
        le_platform: Platform label encoder

    Returns:
        Predicted revenue value
    """
    # Convert to DataFrame
    df = pd.DataFrame([input_data])

    # Transform features
    X = feature_engineer.transform(df, le_source, le_platform)

    # Predict
    prediction = model.predict(X)[0]

    # Ensure non-negative revenue
    return max(0.0, float(prediction))
