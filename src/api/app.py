"""
Revenue Prediction API - Flask microservice for real-time revenue predictions.
"""
import os
import time
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.preprocessing import load_model_artifacts, predict_revenue
from database.db_manager import DatabaseManager
from models.mlflow_manager import MLFlowManager
import mlflow
import mlflow.sklearn

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model artifacts at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'artifacts')
model_from_disk, feature_engineer, le_source, le_platform = load_model_artifacts(MODEL_DIR)

# Initialize database manager
db_manager = DatabaseManager()

# Model metadata
MODEL_METADATA = feature_engineer.metadata

# Global variable for active model
model = None

# Initialize MLFlow and register/load model
def setup_mlflow_model():
    """Register model from disk to MLflow and load it for inference."""
    global model

    try:
        # Get MLflow tracking URI from environment
        mlflow_uri = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5005')

        # Initialize MLFlow manager
        mlflow_manager = MLFlowManager(
            tracking_uri=mlflow_uri,
            experiment_name='revenue_prediction_production'
        )

        model_name = MODEL_METADATA.get('model_name', 'XGBoost')
        registered_model_name = f"revenue_prediction_{model_name.lower().replace(' ', '_')}"

        # Log model to MLflow
        with mlflow.start_run(run_name=f'production_model_{model_name}'):
            # Log model
            mlflow.sklearn.log_model(
                sk_model=model_from_disk,
                artifact_path='model',
                registered_model_name=registered_model_name
            )

            # Log metadata as params
            if MODEL_METADATA.get('best_params'):
                mlflow.log_params(MODEL_METADATA['best_params'])

            # Log metrics from training
            metrics_to_log = {}
            for metric in ['train_mae', 'dev_mae', 'test_mae', 'test_r2']:
                if metric in MODEL_METADATA:
                    metrics_to_log[metric] = MODEL_METADATA[metric]

            if metrics_to_log:
                mlflow.log_metrics(metrics_to_log)

            # Log tags
            mlflow.set_tag('stage', 'production')
            mlflow.set_tag('source', 'disk_load')
            mlflow.set_tag('framework', 'sklearn')
            mlflow.set_tag('split_strategy', MODEL_METADATA.get('training_method', 'unknown'))

            run_id = mlflow.active_run().info.run_id

        # Get the version number from MLflow
        client = mlflow.tracking.MlflowClient()
        model_versions = client.search_model_versions(f"name='{registered_model_name}'")

        # Get the latest version (the one we just registered)
        latest_version = max([int(mv.version) for mv in model_versions]) if model_versions else None

        print(f"✅ Model registered to MLflow: {registered_model_name}")
        print(f"   Run ID: {run_id}")
        print(f"   Version: {latest_version}")
        print(f"   Tracking URI: {mlflow_uri}")

        # Load model from MLflow (version from environment or latest)
        try:
            model_version = os.environ.get('MLFLOW_MODEL_VERSION', 'latest')
            model_uri = f"models:/{registered_model_name}/{model_version}"
            model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Model loaded from MLflow: {model_uri}")
        except Exception as load_error:
            print(f"⚠️  Could not load from MLflow, using disk model: {str(load_error)}")
            model = model_from_disk

        return {
            'run_id': run_id,
            'version': latest_version,
            'model_name': registered_model_name,
            'source': 'mlflow' if model != model_from_disk else 'disk'
        }

    except Exception as e:
        print(f"⚠️  Warning: Could not register model to MLflow: {str(e)}")
        print(f"   Using model from disk as fallback")
        model = model_from_disk
        return None

# Setup MLflow and load model
MLFLOW_INFO = setup_mlflow_model()


@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint.

    Returns:
        JSON response with service status
    """
    mlflow_registered = MLFLOW_INFO is not None

    response = {
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model': MODEL_METADATA.get('model_name', 'Unknown'),
        'mlflow_registered': mlflow_registered,
        'model_source': MLFLOW_INFO.get('source', 'disk') if mlflow_registered else 'disk'
    }

    if mlflow_registered:
        response['version'] = MLFLOW_INFO.get('version', 'Unknown')
        response['mlflow_tracking_uri'] = os.environ.get('MLFLOW_TRACKING_URI', 'http://mlflow:5005')
        response['mlflow_run_id'] = MLFLOW_INFO.get('run_id')
        response['mlflow_model_name'] = MLFLOW_INFO.get('model_name')
    else:
        response['version'] = '1.0.0'

    return jsonify(response), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.

    Expected JSON payload:
    {
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

    Returns:
        JSON response with predicted revenue
    """
    start_time = time.time()

    try:
        # Validate request
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400

        input_data = request.get_json()

        # Validate required fields
        required_fields = [
            'country', 'country_region', 'source', 'platform',
            'device_family', 'os_version', 'event_1', 'event_2', 'event_3'
        ]

        missing_fields = [field for field in required_fields if field not in input_data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}'
            }), 400

        # Make prediction
        predicted_revenue = predict_revenue(
            input_data, model, feature_engineer, le_source, le_platform
        )

        # Calculate inference time
        inference_time = time.time() - start_time

        # Prepare response
        response = {
            'predicted_revenue': round(predicted_revenue, 6),
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }

        # Log to database
        try:
            db_manager.log_prediction(input_data, predicted_revenue, inference_time)
        except Exception as db_error:
            # Don't fail the request if database logging fails
            app.logger.warning(f'Database logging failed: {str(db_error)}')

        return jsonify(response), 200

    except Exception as e:
        app.logger.error(f'Prediction error: {str(e)}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint for multiple users.

    Expected JSON payload:
    {
        "users": [
            {user_data_1},
            {user_data_2},
            ...
        ]
    }

    Returns:
        JSON response with predictions for all users
    """
    start_time = time.time()

    try:
        if not request.is_json:
            return jsonify({
                'error': 'Content-Type must be application/json'
            }), 400

        data = request.get_json()
        users = data.get('users', [])

        if not users or not isinstance(users, list):
            return jsonify({
                'error': 'Expected "users" array in request body'
            }), 400

        predictions = []
        for user_data in users:
            try:
                predicted_revenue = predict_revenue(
                    user_data, model, feature_engineer, le_source, le_platform
                )
                predictions.append({
                    'input': user_data,
                    'predicted_revenue': round(predicted_revenue, 6)
                })
            except Exception as e:
                predictions.append({
                    'input': user_data,
                    'error': str(e)
                })

        inference_time = time.time() - start_time

        return jsonify({
            'predictions': predictions,
            'total_users': len(users),
            'inference_time_ms': round(inference_time * 1000, 2),
            'timestamp': datetime.utcnow().isoformat()
        }), 200

    except Exception as e:
        app.logger.error(f'Batch prediction error: {str(e)}')
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the current model.

    Returns:
        JSON response with model metadata
    """
    return jsonify({
        'model_name': MODEL_METADATA.get('model_name', 'Unknown'),
        'features': MODEL_METADATA.get('feature_cols', []),
        'metrics': {
            'test_mae': MODEL_METADATA.get('test_mae'),
            'test_rmse': MODEL_METADATA.get('test_rmse'),
            'test_r2': MODEL_METADATA.get('test_r2')
        },
        'version': '1.0.0'
    }), 200


@app.route('/stats', methods=['GET'])
def get_stats():
    """
    Get prediction statistics from database.

    Returns:
        JSON response with statistics
    """
    try:
        stats = db_manager.get_prediction_stats()
        return jsonify(stats), 200
    except Exception as e:
        app.logger.error(f'Stats error: {str(e)}')
        return jsonify({
            'error': 'Failed to retrieve statistics',
            'message': str(e)
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({
        'error': 'Internal server error'
    }), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug
    )
