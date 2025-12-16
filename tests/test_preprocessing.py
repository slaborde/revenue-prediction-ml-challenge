"""
Unit tests for preprocessing module.
"""
import unittest
import pandas as pd
import numpy as np
from src.models.preprocessing import FeatureEngineer


class TestFeatureEngineer(unittest.TestCase):
    """Test cases for FeatureEngineer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.metadata = {
            'feature_cols': [
                'event_1', 'event_2', 'event_3', 'total_events',
                'event_1_ratio', 'event_2_ratio', 'event_3_ratio',
                'country_freq', 'country_region_freq', 'device_family_freq',
                'source_encoded', 'platform_encoded', 'country_mean_revenue'
            ],
            'country_mean_revenue': {'es': 0.2, 'us': 0.3, 'ar': 0.15},
            'country_value_counts': {'es': 1000, 'us': 800, 'ar': 500},
            'device_family_value_counts': {'Apple iPhone': 1500, 'Samsung': 800},
            'country_region_value_counts': {'Madrid': 600, 'California': 500}
        }
        self.fe = FeatureEngineer(self.metadata)

    def test_create_derived_features(self):
        """Test creation of derived features."""
        df = pd.DataFrame({
            'event_1': [10, 20, 30],
            'event_2': [5, 10, 15],
            'event_3': [2, None, 8]
        })

        result = self.fe.create_derived_features(df)

        # Check total_events
        self.assertIn('total_events', result.columns)
        self.assertEqual(result['total_events'].iloc[0], 17.0)

        # Check event ratios
        self.assertIn('event_1_ratio', result.columns)
        self.assertAlmostEqual(result['event_1_ratio'].iloc[0], 10/18, places=5)

        # Check null handling
        self.assertEqual(result['event_3'].isna().sum(), 0)

    def test_encode_categorical(self):
        """Test categorical encoding."""
        df = pd.DataFrame({
            'country': ['es', 'us', 'ar'],
            'device_family': ['Apple iPhone', 'Samsung', 'Apple iPhone'],
            'country_region': ['Madrid', 'California', 'Madrid'],
            'platform': ['iOS', 'Android', 'iOS']
        })

        result = self.fe.encode_categorical(df)

        # Check frequency encoding
        self.assertIn('country_freq', result.columns)
        self.assertEqual(result['country_freq'].iloc[0], 1000)

        # Check target encoding
        self.assertIn('country_mean_revenue', result.columns)
        self.assertEqual(result['country_mean_revenue'].iloc[0], 0.2)

        # Check platform normalization
        self.assertTrue(all(result['platform'].str.islower()))

    def test_transform_with_missing_features(self):
        """Test transform handles missing features gracefully."""
        df = pd.DataFrame({
            'event_1': [10],
            'event_2': [5],
            'event_3': [2]
        })

        # Mock encoders
        class MockEncoder:
            classes_ = ['Organic', 'Non-organic']

            def transform(self, x):
                return [0]

        result = self.fe.transform(df, MockEncoder(), MockEncoder())

        # Should have all required features
        self.assertEqual(len(result.columns), len(self.metadata['feature_cols']))

    def test_handles_unknown_categories(self):
        """Test handling of unknown categories in encoders."""
        df = pd.DataFrame({
            'event_1': [10],
            'event_2': [5],
            'event_3': [2],
            'country': ['unknown_country'],
            'device_family': ['unknown_device'],
            'country_region': ['unknown_region'],
            'platform': ['ios'],
            'source': ['Organic']
        })

        class MockEncoder:
            classes_ = ['Organic']

            def transform(self, x):
                return [0] if x[0] == 'Organic' else [-1]

        result = self.fe.transform(df, MockEncoder(), MockEncoder())

        # Should handle unknown country gracefully
        self.assertIsNotNone(result['country_mean_revenue'].iloc[0])


class TestFeatureIntegrity(unittest.TestCase):
    """Test data integrity and edge cases."""

    def test_zero_events(self):
        """Test handling of zero events."""
        metadata = {'feature_cols': []}
        fe = FeatureEngineer(metadata)

        df = pd.DataFrame({
            'event_1': [0],
            'event_2': [0],
            'event_3': [0]
        })

        result = fe.create_derived_features(df)

        # Should not divide by zero
        self.assertEqual(result['event_1_ratio'].iloc[0], 0.0)

    def test_negative_events(self):
        """Test handling of negative events (data quality issue)."""
        metadata = {'feature_cols': []}
        fe = FeatureEngineer(metadata)

        df = pd.DataFrame({
            'event_1': [-5],
            'event_2': [10],
            'event_3': [2]
        })

        result = fe.create_derived_features(df)

        # Should handle negative values
        self.assertEqual(result['total_events'].iloc[0], 7.0)


if __name__ == '__main__':
    unittest.main()
