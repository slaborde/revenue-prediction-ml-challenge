# Revenue Prediction System

Sistema de predicción de revenue para usuarios de juegos móviles durante los primeros 7 días desde la instalación. Implementado como un microservicio Flask de baja latencia con integración de MLFlow y PostgreSQL.

## Estructura de la Documentacion
README.md (este doc) - Guia principal para instalar y entender la solucion, se debe leer primero esta guia antes de leer el resto de los documentos
ENTREGA.md - Overview de la solucion
PROJECT_SUMMARY.txt - Resumen de la solucion implementada
API_DOCS.md - Documentacion detallada del microservicio (API)

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
│   └── model_training_whale_weighted_mlflow.ipynb     # Modelado completo
    └── eda.ipynb     # Notebook con EDA 
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

Para instalar la solucion completa primero se debe correr el notebook de entrenamiento del modelo para generar los artefactos necesarios que necesita el micro servicio para servir las predicciones. En caso de no querer hacerlo se puede saltear el paso ya que los artefactos estan previamente generados en la carpeta correspondiente. 

Adicionalmete se incluyo un paso **solo con fines de simplificar la instalacion del challenge, no forma parte de la solucion productiva** donde si los artefactos no se encuentran presentes en la carpeta esperada, previo a iniciar el micro servicio se corre (via script) automaticamente el entrenamiento del modelo para generar los artefactos necesarios.

### Paso 1: Entrenamiento del Modelo

Para examinar la parte de analisis y entrenamiento del modelo se puede setear un venv de python, activarlo e instalar las librerias de requeriments luego se pueden correr los notebooks de analisis exploratorio eda.ipynb y el notebook de entrenamiento del modelo model_training_whale_weighted_mlflow.ipynb

```bash
# Crear entorno virtual
python -m venv venv_regal
source venv_regal/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias

# Para el caso de Mac Os, se recomienda correr esta linea (instala una version pre-compilada) previo a instalar requirements en otro caso saltear la linea
# ya que puede fallar la instalacion de lightGBM porque requiere compilacion y esto puede dar problemas
pip install lightgbm==4.6.0 --no-build-isolation # Solo para Mac Os en otro caso saltar el paso

# Instalar resto de dependencias
pip install -r requirements.txt

# Ejecutar notebook para entrenar modelo
jupyter notebook notebooks/model_training_whale_weighted_mlflow.ipynb
```
#### Uso

#### Entrenar el Modelo
1. Abrir el notebook `notebooks/model_training_whale_weighted_mlflow.ipynb`
2. Ejecutar todas las celdas
3. El modelo y artefactos se guardarán en `src/models/artifacts/`

### Instalar el MicroServicio (API)
```bash
# Construir y levantar todos los servicios (si se desea se puede ejecutar docker-compose up --build en lugar de docker build y luego docker up)
docker-compose build --no-cache
docker-compose up -d
# Opcional: Para chequear finalizacion correcta se pueden examinar los logs del micro servicio
docker-compose logs -f api
# La API estará disponible en http://localhost:5001 
# Para chequear su funcionamiento ejecute http://localhost:5001/health abajo hay mas ejemplos de su uso
# MLFlow UI en http://localhost:5005 aqui se puede verificar que esta levantado el server de MLFlow y esta el modelo registrado
```

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
    "timestamp": "2025-12-17T15:30:45.123456",
    "model": "XGBoost",
    "mlflow_registered": true,
    "model_source": "mlflow",
    "version": 1,
    "mlflow_tracking_uri": "http://mlflow:5005",
    "mlflow_run_id": "abc123def456...",
    "mlflow_model_name": "revenue_prediction_xgboost"
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
    "features": [
      "country_mean_revenue",
      "event_1",
      "event_2",
      "event_3",
      "country_value_counts",
      "device_family_value_counts",
      "country_region_value_counts",
      "source_encoded",
      "platform_encoded",
      "os_version_major",
      "os_version_minor",
      "event_1_log",
      "event_2_log"
    ],
    "metrics": {
      "test_mae": 15.82,
      "test_rmse": 24.66,
      "test_r2": 0.909
    },
    "version": "1.0.0"
  }
```

### GET /stats
Estadísticas de predicciones realizadas.

**Response:**
```json
{
    "avg_inference_time_ms": 157.39834308624268,
    "avg_predicted_revenue": 22.16754913330078,
    "first_prediction": "Wed, 17 Dec 2025 01:55:51 GMT",
    "last_prediction": "Wed, 17 Dec 2025 16:31:05 GMT",
    "max_predicted_revenue": 22.16754913330078,
    "min_predicted_revenue": 22.16754913330078,
    "platform_distribution": [
      {
        "count": 2,
        "platform": "iOS"
      }
    ],
    "top_countries": [
      {
        "count": 2,
        "country": "es"
      }
    ],
    "total_predictions": 2
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
- XGBoost (seleccionado)
- LightGBM 

### Métricas de Evaluación

El modelo final fue evaluado usando:
- **MAE (Mean Absolute Error):** Métrica principal
  Porque ?
    1. Interpretable en términos de negocio: MAE=15.82 significa que en promedio nos equivocamos por $15.82 en la predicción de revenue, directamente entendible para stakeholders.
    2. Robusta a outliers (whales): A diferencia de RMSE/MSE que penalizan cuadráticamente, MAE trata todos los errores linealmente, evitando que usuarios de alto revenue (whales) dominen la optimización del modelo.
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
PORT=5001
DEBUG=False
MLFLOW_MODEL_VERSION=latest
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
1. Mejor Performance en Métricas

  - R² = 0.909: Explica el 90.9% de la varianza en revenue
  - MAE = 15.82: Error absoluto medio más bajo que otros modelos
  - RMSE = 24.66: Mejor predicción que Random Forest y LightGBM

  2. Manejo Excelente de Whales (High-Value Users)

  Durante el análisis exploratorio descubrimos que el 99.6% del revenue viene de solo el 15% de usuarios (Perú y otros países con whales). XGBoost:
  - Captura bien patrones no lineales de comportamiento de whales
  - Maneja efectivamente outliers (usuarios con revenue muy alto)
  - Usa gradient boosting que se enfoca en errores difíciles (como predecir whales)

  3. Robustez con Features de Comportamiento

  - Maneja bien event_1, event_2, event_3 (eventos de usuario)
  - Utiliza efectivamente target encoding (country_mean_revenue)
  - No requiere normalización de features

  4. Ventajas Técnicas sobre LightGBM y Random Forest

  vs LightGBM:
  - Similar en velocidad pero mejor accuracy en nuestro dataset
  - Más estable con whale-weighted split

  vs Random Forest:
  - Mejor con datos desbalanceados (whales vs no-whales)
  - Gradient boosting > bagging para este caso

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