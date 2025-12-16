# Revenue Prediction API - Documentation

## Base URL

```
http://localhost:5001
```

## Authentication

Currently, the API does not require authentication. In a production environment, implement API keys or OAuth2.

## Endpoints

### 1. Health Check

Check the health status of the API.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00.000Z",
  "model": "LightGBM",
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: Service is healthy

**Example:**
```bash
curl http://localhost:5001/health
```

---

### 2. Single Prediction

Predict revenue for a single user.

**Endpoint:** `POST /predict`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
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
```

**Field Descriptions:**
- `country` (string, required): ISO country code
- `country_region` (string, required): Region/province
- `source` (string, required): User acquisition source (Organic/Non-organic)
- `platform` (string, required): Platform (iOS/Android)
- `device_family` (string, required): Device model
- `os_version` (string, required): Operating system version
- `event_1` (number, required): Count of event type 1
- `event_2` (number, required): Count of event type 2
- `event_3` (number, required): Count of event type 3 (can be null)

**Response:**
```json
{
  "predicted_revenue": 0.234567,
  "inference_time_ms": 12.34,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Status Codes:**
- `200 OK`: Prediction successful
- `400 Bad Request`: Invalid input (missing fields, wrong format)
- `500 Internal Server Error`: Server error

**Example:**
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "country": "es",
    "country_region": "Madrid",
    "source": "Organic",
    "platform": "iOS",
    "device_family": "Apple iPhone",
    "os_version": "14.4",
    "event_1": 100,
    "event_2": 50,
    "event_3": 10.0
  }'
```

**Python Example:**
```python
import requests

url = "http://localhost:5001/predict"
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

response = requests.post(url, json=data)
print(response.json())
```

---

### 3. Batch Prediction

Predict revenue for multiple users in a single request.

**Endpoint:** `POST /batch_predict`

**Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
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
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "input": {...},
      "predicted_revenue": 0.234567
    },
    {
      "input": {...},
      "predicted_revenue": 0.187654
    }
  ],
  "total_users": 2,
  "inference_time_ms": 25.67,
  "timestamp": "2024-01-15T10:30:00.000Z"
}
```

**Status Codes:**
- `200 OK`: Predictions successful
- `400 Bad Request`: Invalid input
- `500 Internal Server Error`: Server error

**Example:**
```bash
curl -X POST http://localhost:5001/batch_predict \
  -H "Content-Type: application/json" \
  -d '{
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
      }
    ]
  }'
```

---

### 4. Model Information

Get information about the current model in production.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_name": "LightGBM",
  "features": [
    "event_1",
    "event_2",
    "event_3",
    "total_events",
    "event_1_ratio",
    "event_2_ratio",
    "event_3_ratio",
    "country_freq",
    "country_region_freq",
    "device_family_freq",
    "source_encoded",
    "platform_encoded",
    "country_mean_revenue"
  ],
  "metrics": {
    "test_mae": 0.012345,
    "test_rmse": 0.045678,
    "test_r2": 0.89
  },
  "version": "1.0.0"
}
```

**Status Codes:**
- `200 OK`: Success

**Example:**
```bash
curl http://localhost:5001/model/info
```

---

### 5. Prediction Statistics

Get statistics about predictions made by the system.

**Endpoint:** `GET /stats`

**Response:**
```json
{
  "total_predictions": 1234,
  "avg_predicted_revenue": 0.234567,
  "min_predicted_revenue": 0.0,
  "max_predicted_revenue": 1.5,
  "avg_inference_time_ms": 15.6,
  "first_prediction": "2024-01-10T08:00:00.000Z",
  "last_prediction": "2024-01-15T10:30:00.000Z",
  "top_countries": [
    {"country": "es", "count": 500},
    {"country": "us", "count": 300}
  ],
  "platform_distribution": [
    {"platform": "ios", "count": 700},
    {"platform": "android", "count": 534}
  ]
}
```

**Status Codes:**
- `200 OK`: Success
- `500 Internal Server Error`: Database error

**Example:**
```bash
curl http://localhost:5001/stats
```

---

## Error Responses

All endpoints return errors in the following format:

```json
{
  "error": "Error type",
  "message": "Detailed error message"
}
```

### Common Error Codes:

- `400 Bad Request`: Invalid input data
  ```json
  {
    "error": "Missing required fields: event_1, event_2"
  }
  ```

- `404 Not Found`: Endpoint not found
  ```json
  {
    "error": "Endpoint not found"
  }
  ```

- `500 Internal Server Error`: Server error
  ```json
  {
    "error": "Internal server error",
    "message": "Model prediction failed"
  }
  ```

---

## Performance Considerations

### Latency

- Single prediction: ~10-20ms
- Batch prediction (100 users): ~200-300ms

### Throughput

The API can handle approximately:
- 50+ predictions/second (single)
- 10+ batch requests/second (100 users each)

### Best Practices

1. **Use batch predictions** when predicting for multiple users
2. **Implement client-side timeout** (recommend 5 seconds)
3. **Handle errors gracefully** with retry logic
4. **Cache predictions** when appropriate (if user data doesn't change)

---

## Rate Limiting

Currently, there are no rate limits. In production, implement:
- Per-IP rate limiting
- API key-based quotas

---

## Monitoring

The API logs all predictions to PostgreSQL. Monitor:
- Prediction volume
- Inference time
- Error rates
- Input distribution

Access logs through the `/stats` endpoint or directly from the database.

---

## Data Privacy

- User IDs are anonymized
- Input data is logged for monitoring purposes
- Implement data retention policies for compliance

---

## Support

For issues or questions:
1. Check logs: `docker-compose logs -f api`
2. Verify model health: `GET /health`
3. Review prediction stats: `GET /stats`
