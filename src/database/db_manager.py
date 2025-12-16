"""
Database manager for logging predictions.
"""
import os
import json
from datetime import datetime
from typing import Dict, Any
import psycopg2
from psycopg2.extras import RealDictCursor


class DatabaseManager:
    """
    Manages database connections and operations for prediction logging.
    """

    def __init__(self):
        """Initialize database connection parameters from environment variables."""
        self.host = os.environ.get('DB_HOST', 'localhost')
        self.port = os.environ.get('DB_PORT', '5432')
        self.database = os.environ.get('DB_NAME', 'revenue_predictions')
        self.user = os.environ.get('DB_USER', 'postgres')
        self.password = os.environ.get('DB_PASSWORD', 'postgres')

        # Create databases if they don't exist
        self._create_databases()

        # Create table if it doesn't exist
        self._create_table()

    def _get_connection(self, database=None):
        """
        Get database connection.

        Args:
            database: Database name to connect to (defaults to self.database)

        Returns:
            Database connection object
        """
        try:
            conn = psycopg2.connect(
                host=self.host,
                port=self.port,
                database=database or self.database,
                user=self.user,
                password=self.password
            )
            return conn
        except psycopg2.OperationalError:
            # If database connection fails, return None
            # This allows the API to work without database
            return None

    def _create_databases(self):
        """Create required databases if they don't exist."""
        try:
            # Connect to default 'postgres' database to create other databases
            conn = self._get_connection(database='postgres')
            if conn is None:
                return

            # Set autocommit to create databases
            conn.autocommit = True
            cursor = conn.cursor()

            # List of databases to create
            databases = ['revenue_predictions', 'mlflow']

            for db_name in databases:
                # Check if database exists
                cursor.execute(
                    "SELECT 1 FROM pg_database WHERE datname = %s",
                    (db_name,)
                )
                exists = cursor.fetchone()

                if not exists:
                    cursor.execute(f'CREATE DATABASE {db_name}')
                    print(f"✅ Database '{db_name}' created successfully")
                else:
                    print(f"ℹ️  Database '{db_name}' already exists")

            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error creating databases: {str(e)}")

    def _create_table(self):
        """Create predictions table if it doesn't exist."""
        try:
            conn = self._get_connection()
            if conn is None:
                return

            cursor = conn.cursor()

            create_table_query = """
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                country VARCHAR(50),
                country_region VARCHAR(100),
                source VARCHAR(50),
                platform VARCHAR(50),
                device_family VARCHAR(100),
                os_version VARCHAR(50),
                event_1 FLOAT,
                event_2 FLOAT,
                event_3 FLOAT,
                predicted_revenue FLOAT NOT NULL,
                inference_time_ms FLOAT,
                input_data JSONB
            );

            CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp);
            CREATE INDEX IF NOT EXISTS idx_country ON predictions(country);
            CREATE INDEX IF NOT EXISTS idx_platform ON predictions(platform);
            """

            cursor.execute(create_table_query)
            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error creating table: {str(e)}")

    def log_prediction(self, input_data: Dict[str, Any],
                      predicted_revenue: float,
                      inference_time: float):
        """
        Log a prediction to the database.

        Args:
            input_data: Input features used for prediction
            predicted_revenue: Predicted revenue value
            inference_time: Time taken for inference in seconds
        """
        try:
            conn = self._get_connection()
            if conn is None:
                return

            cursor = conn.cursor()

            insert_query = """
            INSERT INTO predictions (
                country, country_region, source, platform, device_family,
                os_version, event_1, event_2, event_3,
                predicted_revenue, inference_time_ms, input_data
            ) VALUES (
                %(country)s, %(country_region)s, %(source)s, %(platform)s,
                %(device_family)s, %(os_version)s, %(event_1)s, %(event_2)s,
                %(event_3)s, %(predicted_revenue)s, %(inference_time_ms)s,
                %(input_data)s
            )
            """

            params = {
                'country': input_data.get('country'),
                'country_region': input_data.get('country_region'),
                'source': input_data.get('source'),
                'platform': input_data.get('platform'),
                'device_family': input_data.get('device_family'),
                'os_version': input_data.get('os_version'),
                'event_1': input_data.get('event_1'),
                'event_2': input_data.get('event_2'),
                'event_3': input_data.get('event_3'),
                'predicted_revenue': predicted_revenue,
                'inference_time_ms': inference_time * 1000,
                'input_data': json.dumps(input_data)
            }

            cursor.execute(insert_query, params)
            conn.commit()
            cursor.close()
            conn.close()

        except Exception as e:
            print(f"Error logging prediction: {str(e)}")
            raise

    def get_prediction_stats(self) -> Dict[str, Any]:
        """
        Get statistics about predictions.

        Returns:
            Dictionary with prediction statistics
        """
        try:
            conn = self._get_connection()
            if conn is None:
                return {
                    'error': 'Database connection not available',
                    'total_predictions': 0
                }

            cursor = conn.cursor(cursor_factory=RealDictCursor)

            stats_query = """
            SELECT
                COUNT(*) as total_predictions,
                AVG(predicted_revenue) as avg_predicted_revenue,
                MIN(predicted_revenue) as min_predicted_revenue,
                MAX(predicted_revenue) as max_predicted_revenue,
                AVG(inference_time_ms) as avg_inference_time_ms,
                MIN(timestamp) as first_prediction,
                MAX(timestamp) as last_prediction
            FROM predictions
            """

            cursor.execute(stats_query)
            stats = dict(cursor.fetchone())

            # Get top countries
            cursor.execute("""
                SELECT country, COUNT(*) as count
                FROM predictions
                WHERE country IS NOT NULL
                GROUP BY country
                ORDER BY count DESC
                LIMIT 10
            """)
            top_countries = [dict(row) for row in cursor.fetchall()]
            stats['top_countries'] = top_countries

            # Get platform distribution
            cursor.execute("""
                SELECT platform, COUNT(*) as count
                FROM predictions
                WHERE platform IS NOT NULL
                GROUP BY platform
                ORDER BY count DESC
            """)
            platform_dist = [dict(row) for row in cursor.fetchall()]
            stats['platform_distribution'] = platform_dist

            cursor.close()
            conn.close()

            return stats

        except Exception as e:
            return {
                'error': str(e),
                'total_predictions': 0
            }
