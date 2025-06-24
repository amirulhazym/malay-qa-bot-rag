#!/bin/bash

# -- Make the script "fail loud" --
# This command ensures that if any single command fails, the whole script exits immediately.
set -e

# -- Add Execute Permissions --
# This ensures the server can run this file as a script.
chmod +x setup.sh

echo "--- setup.sh script has started ---"

# --- Start the Backend with Logging ---
echo "Attempting to start backend server..."
uvicorn v2_multilingual_api.backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

echo "Waiting for backend to initialize (20 seconds)..."
sleep 20 

# --- Health Check ---
echo "Performing health check on the backend..."
if ! pgrep -f "uvicorn v2_multilingual_api.backend.main:app"; then
    echo "--- !!! BACKEND FAILED TO START !!! ---"
    echo "--- Displaying contents of backend.log: ---"
    cat backend.log
    exit 1
else
    echo "Backend health check PASSED. Process is running."
fi

# --- Launch the Frontend ---
echo "Starting frontend server..."
streamlit run v2_multilingual_api.frontend/app.py --server.port 7860 --server.address 0.0.0.0

echo "--- setup.sh script has finished ---"