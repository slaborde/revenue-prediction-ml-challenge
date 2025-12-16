"""
Unit tests for Flask API.
"""
import unittest
import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.api.app import app


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""

    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

        # Sample valid input
        self.valid_input = {
            "country": "es",
            "country_region": "Madrid",
            "source": "Organic",
            "platform": "iOS",
            "device_family": "Apple iPhone",
            "os_version": "14.4",
            "event_1": 100,
            "event_2": 50,
            "event_3": 10.0
        }

    def test_health_check(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'healthy')
        self.assertIn('model', data)
        self.assertIn('timestamp', data)

    def test_predict_endpoint_success(self):
        """Test successful prediction."""
        response = self.client.post(
            '/predict',
            data=json.dumps(self.valid_input),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('predicted_revenue', data)
        self.assertIn('inference_time_ms', data)
        self.assertIn('timestamp', data)

        # Revenue should be non-negative
        self.assertGreaterEqual(data['predicted_revenue'], 0)

    def test_predict_missing_fields(self):
        """Test prediction with missing required fields."""
        incomplete_input = {
            "country": "es",
            "platform": "iOS"
        }

        response = self.client.post(
            '/predict',
            data=json.dumps(incomplete_input),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_predict_invalid_content_type(self):
        """Test prediction with invalid content type."""
        response = self.client.post(
            '/predict',
            data="not json",
            content_type='text/plain'
        )

        self.assertEqual(response.status_code, 400)

        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_batch_predict(self):
        """Test batch prediction endpoint."""
        batch_input = {
            "users": [
                self.valid_input,
                {**self.valid_input, "country": "us"},
                {**self.valid_input, "platform": "Android"}
            ]
        }

        response = self.client.post(
            '/batch_predict',
            data=json.dumps(batch_input),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('predictions', data)
        self.assertIn('total_users', data)
        self.assertEqual(data['total_users'], 3)
        self.assertEqual(len(data['predictions']), 3)

    def test_batch_predict_empty_list(self):
        """Test batch prediction with empty user list."""
        batch_input = {"users": []}

        response = self.client.post(
            '/batch_predict',
            data=json.dumps(batch_input),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 400)

    def test_model_info_endpoint(self):
        """Test model info endpoint."""
        response = self.client.get('/model/info')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('model_name', data)
        self.assertIn('features', data)
        self.assertIn('metrics', data)
        self.assertIn('version', data)

    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = self.client.get('/stats')
        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        self.assertIn('total_predictions', data)

    def test_404_error(self):
        """Test 404 error handling."""
        response = self.client.get('/nonexistent')
        self.assertEqual(response.status_code, 404)

        data = json.loads(response.data)
        self.assertIn('error', data)


class TestPredictionLogic(unittest.TestCase):
    """Test prediction logic and edge cases."""

    def setUp(self):
        """Set up test client."""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()

    def test_zero_events(self):
        """Test prediction with zero events."""
        input_data = {
            "country": "es",
            "country_region": "Madrid",
            "source": "Organic",
            "platform": "iOS",
            "device_family": "Apple iPhone",
            "os_version": "14.4",
            "event_1": 0,
            "event_2": 0,
            "event_3": 0
        }

        response = self.client.post(
            '/predict',
            data=json.dumps(input_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        # Should return low revenue for zero events
        self.assertGreaterEqual(data['predicted_revenue'], 0)

    def test_high_events(self):
        """Test prediction with high event counts."""
        input_data = {
            "country": "es",
            "country_region": "Madrid",
            "source": "Organic",
            "platform": "iOS",
            "device_family": "Apple iPhone",
            "os_version": "14.4",
            "event_1": 10000,
            "event_2": 5000,
            "event_3": 2000
        }

        response = self.client.post(
            '/predict',
            data=json.dumps(input_data),
            content_type='application/json'
        )

        self.assertEqual(response.status_code, 200)

        data = json.loads(response.data)
        # Should handle large values
        self.assertGreaterEqual(data['predicted_revenue'], 0)


if __name__ == '__main__':
    unittest.main()
