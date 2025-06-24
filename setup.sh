#!/bin/bash
# This script MUST be executable: chmod +x setup.sh

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- [setup.sh] Starting backend server... ---"
uvicorn v2_multilingual_api.backend.main:app --host 0.0.0.0 --port 8000 &

echo "--- [setup.sh] Waiting 20 seconds for backend to initialize models... ---"
sleep 20

echo "--- [setup.sh] Performing health check on the backend... ---"
# Use curl to check if the backend's root endpoint is responding
if curl --fail http://127.0.0.1:8000; then
    echo "--- [setup.sh] Backend health check PASSED. Process is running. ---"
else
    echo "--- [setup.sh] !!! BACKEND FAILED TO START OR IS UNRESPONSIVE !!! ---"
    exit 1
fi

echo "--- [setup.sh] Starting frontend server... ---"
streamlit run v2_multilingual_api/frontend/app.py --server.port 7860 --server.address 0.0.0.0