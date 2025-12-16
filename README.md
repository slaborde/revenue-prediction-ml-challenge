# Revenue Prediction System

Sistema de predicción de revenue para usuarios de juegos móviles durante los primeros 7 días desde la instalación. Implementado como un microservicio Flask de baja latencia con integración de MLFlow y PostgreSQL.

## Tabla de Contenidos

- [Descripción](#descripción)
- [Arquitectura](#arquitectura)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [API Endpoints](#api-endpoints)
- [Modelo de ML](#modelo-de-ml)
- [Testing](#testing)
- [Deployment con Docker](#deployment-con-docker)
- [MLFlow Integration](#mlflow-integration)

## Descripción

Este proyecto implementa una solución completa de Machine Learning para predecir el revenue que generará un usuario en sus primeros 7 días de uso. El sistema incluye:

- Análisis exploratorio de datos (EDA) completo
- Pipeline de feature engineering
- Múltiples modelos de ML evaluados
- API REST de baja latencia
- Logging de predicciones en base de datos PostgreSQL
- Tracking de modelos con MLFlow
- Deployment containerizado con Docker

## Arquitectura

```
┌─────────────┐      ┌──────────────┐      ┌────────────┐
│   Cliente   │─────▶│  Flask API   │─────▶│  Modelo ML │
└─────────────┘      └──────────────┘      └────────────┘
                           │
                           ├──────▶ PostgreSQL (Logs)
                           │
                           └──────▶ MLFlow (Tracking)
```

## Estructura del Proyecto

```
regal_cinemas/
├── data/
│   └── dataset.csv                 # Dataset original
├── notebooks/
│   └── model_development.ipynb     # Notebook con EDA y modelado completo
├── src/
│   ├── api/
│   │   ├── __init__.py
│   │   └── app.py                  # Flask API
│   ├── models/
│   │   ├── __init__.py
│   │   ├── preprocessing.py        # Feature engineering
│   │   ├── mlflow_manager.py       # Integración MLFlow
│   │   └── artifacts/              # Modelos y artefactos guardados
│   └── database/
│       ├── __init__.py
│       └── db_manager.py           # Gestión de base de datos
├── tests/
│   ├── __init__.py
│   ├── test_api.py                 # Tests del API
│   └── test_preprocessing.py      # Tests de preprocessing
├── docker/
├── Dockerfile                      # Imagen Docker del API
├── docker-compose.yml              # Orquestación de servicios
├── requirements.txt                # Dependencias Python
└── README.md                       # Este archivo
```

## Instalación

### Desde 0 (Docker)

```bash
# Construir y levantar todos los servicios
docker-compose build --no-cache
docker-compose up -d

Una vez que se levanten todos los servicios, si los artefactos del modelo no se encuentran dentro de la carpeta src/models/artifacts al iniciar el contenedor de la api se correra un script que entrenara el modelo generando los artefactos necesarios, se debe esperar a que termine para que luego se inicie el micro servicio, el servidor de MLFlow y se registre el modelo entrenado en la registry del servidor de MLFlow.

Para chequear finalizacion correcta se pueden examinar los logs del micro servicio
docker-compose logs -f api

# La API estará disponible en http://localhost:5001
# Para chequear su funcionamiento ejecute http://localhost:5001/health
# MLFlow UI en http://localhost:5005

```
### Entrenamiento del Modelo

Para examinar la parte de analisis y entrenamiento del modelo se puede setear un venv de python, activarlo e instalar las librerias de requeriments luego se pueden correr los notebooks de analisis exploratorio eda.ipynb y el notebook de entrenamiento del modelo model_training_whale_weighted_mlflow.ipynb, si se corre este ultimo ya quedaron generados los artefactos del modelo, de esta forma se usaran estos artefactos y no sera necesario correr el entrenamiento dentro del contenedor

```bash
# Crear entorno virtual
python -m venv venv_regal
source venv_regal/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebook para entrenar modelo
jupyter notebook notebooks/model_training_whale_weighted_mlflow.ipynb
```
## Uso

### Entrenar el Modelo
1. Abrir el notebook `notebooks/model_training_whale_weighted_mlflow.ipynb` y `notebooks/eda.ipynb.ipynb`
2. Ejecutar todas las celdas
3. El modelo y artefactos se guardarán en `src/models/artifacts/`

### Hacer Predicciones

```bash
# Health check
curl http://localhost:5001/health

# Predicción individual 1
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

# Predicción individual 2
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "country": "pe",
    "country_region": "Lima",
    "source": "Organic",
    "platform": "iOS",
    "device_family": "Apple iPhone",
    "os_version": "14.4",
    "event_1": 100,
    "event_2": 100,
    "event_3": 0
  }'
```

## API Endpoints

### GET /health
Health check del servicio.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00",
  "model": "LightGBM",
  "version": "1.0.0"
}
```

### POST /predict
Predecir revenue para un usuario.

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

**Response:**
```json
{
  "predicted_revenue": 0.234567,
  "inference_time_ms": 12.34,
  "timestamp": "2024-01-15T10:30:00"
}
```

### POST /batch_predict
Predicciones en batch para múltiples usuarios.

**Request Body:**
```json
{
  "users": [
    {user_data_1},
    {user_data_2},
    ...
  ]
}
```

### GET /model/info

curl http://localhost:5001/model/info

Información sobre el modelo en producción.

**Response:**
```json
{
  "model_name": "XGBoost",
  "features": ["event_1", "event_2", ...],
  "metrics": {
    "test_mae": 0.0123,
    "test_rmse": 0.0456,
    "test_r2": 0.89
  },
  "version": "1.0.0"
}
```

### GET /stats
Estadísticas de predicciones realizadas.

**Response:**
```json
{
  "total_predictions": 1234,
  "avg_predicted_revenue": 0.234,
  "avg_inference_time_ms": 15.6,
  "top_countries": [...],
  "platform_distribution": [...]
}
```

## Modelo de ML

### Features Utilizadas

El modelo utiliza las siguientes características:

- **Eventos del usuario:** `event_1`, `event_2`, `event_3`
- **Features derivadas:** `total_events`, ratios de eventos
- **Información geográfica:** País, región (frequency encoding)
- **Información técnica:** Platform, device family, OS version
- **Target encoding:** Revenue promedio por país

### Modelos Evaluados

Se evaluaron múltiples modelos:
- Ridge Regression
- Lasso Regression
- Random Forest
- Gradient Boosting
- XGBoost
- LightGBM (seleccionado)

### Métricas de Evaluación

El modelo final fue evaluado usando:
- **MAE (Mean Absolute Error):** Métrica principal
- **RMSE (Root Mean Squared Error):** Error cuadrático
- **R² Score:** Capacidad explicativa

### Performance

El sistema está optimizado para baja latencia:
- Tiempo de inferencia típico: < 20ms
- Throughput: > 50 predicciones/segundo

## Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Test específico con coverage
pytest tests/test_api.py -v --cov=src

# Test de preprocessing
pytest tests/test_preprocessing.py -v
```

### Tests Implementados

- **test_api.py:** Tests de endpoints, validaciones, edge cases
- **test_preprocessing.py:** Tests de feature engineering, encoding, transformaciones

## Deployment con Docker

### Servicios Incluidos

1. **PostgreSQL:** Base de datos para logging de predicciones
2. **MLFlow:** Servidor de tracking de modelos
3. **API:** Microservicio Flask de predicción

### Comandos Docker

```bash
# Levantar todos los servicios
docker-compose up -d

# Ver logs
docker-compose logs -f api

# Detener servicios
docker-compose down

# Rebuild sin cache
docker-compose build --no-cache
```

### Variables de Entorno

Las siguientes variables pueden configurarse en `docker-compose.yml`:

```yaml
DB_HOST=db
DB_PORT=5432
DB_NAME=revenue_predictions
DB_USER=postgres
DB_PASSWORD=postgres
MLFLOW_TRACKING_URI=http://mlflow:5005
PORT=5000
DEBUG=False
```

## MLFlow Integration

### Tracking de Modelos

MLFlow se utiliza para:
- Tracking de experimentos y métricas
- Versionado de modelos
- Registry de modelos
- Comparación de performance

### Acceder a MLFlow UI

Después de levantar los servicios:

```
http://localhost:5005
```

### Uso Programático

```python
from src.models.mlflow_manager import MLFlowManager

# Inicializar manager
mlflow_manager = MLFlowManager(
    tracking_uri="http://localhost:5005",
    experiment_name="revenue_prediction"
)

# Log training run
run_id = mlflow_manager.log_model_training(
    model=trained_model,
    model_name="LightGBM_v1",
    params=model.get_params(),
    metrics={"test_mae": 0.0123},
    artifacts_dir="./artifacts"
)

# Load model from registry
model = mlflow_manager.load_model("LightGBM_v1", version="latest")
```

## Estructura de Base de Datos

### Tabla: predictions

```sql
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
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
```

## Decisiones Técnicas

### Feature Engineering

- **Frequency Encoding:** Para variables de alta cardinalidad (country, device_family)
- **Target Encoding:** Para country (con validación cruzada implícita)
- **Ratios de eventos:** Para capturar patrones de comportamiento
- **Normalización de categorías:** Manejo de inconsistencias (iOS/ios)

### Selección de Modelo

Se seleccionó XGBoost por:
- Mejor performance en métricas de evaluación
- Velocidad de inferencia
- Robustez a outliers
- Manejo nativo de features categóricas

### Optimizaciones de Performance

- **Modelo precompilado:** Cargado una sola vez al iniciar el servicio
- **Feature engineering optimizado:** Sin operaciones costosas
- **Encoders precalculados:** Transformaciones rápidas
- **Sin I/O en inferencia:** Todo en memoria

## Mejoras Futuras

1. **Modelo:**
   - A/B testing de modelos
   - Reentrenamiento automático
   - Detección de drift

2. **API:**
   - Caché de predicciones frecuentes
   - Rate limiting
   - Autenticación/autorización

3. **Monitoreo:**
   - Métricas de Prometheus
   - Dashboards de Grafana
   - Alertas de performance

4. **Data:**
   - Pipeline de ETL automatizado
   - Validación de datos de entrada
   - Feature store