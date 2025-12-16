"""
Example script to test the Revenue Prediction API.
"""
import requests
import json
import time


def test_health():
    """Test health endpoint."""
    print("Testing /health endpoint...")
    response = requests.get("http://localhost:5000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_predict():
    """Test single prediction endpoint."""
    print("Testing /predict endpoint...")

    data = {
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

    response = requests.post(
        "http://localhost:5000/predict",
        json=data
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_batch_predict():
    """Test batch prediction endpoint."""
    print("Testing /batch_predict endpoint...")

    data = {
        "users": [
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
            },
            {
                "country": "us",
                "country_region": "California",
                "source": "Non-organic",
                "platform": "Android",
                "device_family": "Samsung Galaxy",
                "os_version": "11.0",
                "event_1": 80,
                "event_2": 40,
                "event_3": 5.0
            },
            {
                "country": "ar",
                "country_region": "Buenos Aires",
                "source": "Organic",
                "platform": "iOS",
                "device_family": "Apple iPad",
                "os_version": "13.5",
                "event_1": 120,
                "event_2": 60,
                "event_3": 15.0
            }
        ]
    }

    response = requests.post(
        "http://localhost:5000/batch_predict",
        json=data
    )

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_model_info():
    """Test model info endpoint."""
    print("Testing /model/info endpoint...")

    response = requests.get("http://localhost:5000/model/info")

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def test_stats():
    """Test stats endpoint."""
    print("Testing /stats endpoint...")

    response = requests.get("http://localhost:5000/stats")

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}\n")


def benchmark_latency(n_requests=100):
    """Benchmark prediction latency."""
    print(f"Benchmarking latency with {n_requests} requests...")

    data = {
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

    latencies = []

    for i in range(n_requests):
        start = time.time()
        response = requests.post("http://localhost:5000/predict", json=data)
        latency = (time.time() - start) * 1000  # Convert to ms

        if response.status_code == 200:
            latencies.append(latency)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i + 1}/{n_requests}")

    if latencies:
        print(f"\nLatency Statistics:")
        print(f"  Mean: {sum(latencies) / len(latencies):.2f} ms")
        print(f"  Min: {min(latencies):.2f} ms")
        print(f"  Max: {max(latencies):.2f} ms")
        print(f"  P50: {sorted(latencies)[len(latencies) // 2]:.2f} ms")
        print(f"  P95: {sorted(latencies)[int(len(latencies) * 0.95)]:.2f} ms")
        print(f"  P99: {sorted(latencies)[int(len(latencies) * 0.99)]:.2f} ms\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Revenue Prediction API - Test Suite")
    print("=" * 60 + "\n")

    try:
        # Basic tests
        test_health()
        test_predict()
        test_batch_predict()
        test_model_info()
        test_stats()

        # Performance benchmark
        benchmark_latency(n_requests=50)

        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("ERROR: Could not connect to API.")
        print("Make sure the API is running at http://localhost:5000")
    except Exception as e:
        print(f"ERROR: {str(e)}")


if __name__ == "__main__":
    main()
