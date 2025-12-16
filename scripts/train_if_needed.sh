#!/bin/bash

# Script to train model if artifacts don't exist

ARTIFACTS_DIR="/app/src/models/artifacts"
MODEL_FILE="$ARTIFACTS_DIR/model.pkl"
NOTEBOOK="/app/notebooks/model_training_whale_weighted.ipynb"

echo "Checking if model artifacts exist..."

if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model artifacts found at $MODEL_FILE"
    echo "   Skipping training..."
else
    echo "⚠️  Model artifacts NOT found"
    echo "   Running training notebook: $NOTEBOOK"
    echo ""

    # Execute notebook using nbconvert
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=3600 \
        --output /tmp/executed_notebook.ipynb \
        "$NOTEBOOK"

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Training completed successfully!"
        echo "   Model artifacts saved to: $ARTIFACTS_DIR"
    else
        echo ""
        echo "❌ Training failed!"
        echo "   Starting API anyway (will fail if no model exists)"
    fi
fi

echo ""
echo "Starting API..."

# Start the API
exec python -m src.api.app
