#!/bin/bash

# This line tells the server that this is a shell script

# --- Start the Backend with Logging ---
echo "Starting backend server and logging to backend.log..."
# The > backend.log part redirects all normal output to a file named backend.log
# The 2>&1 part redirects all error output to the same file.
# The final '&' still runs it in the background.
uvicorn v2_multilingual_api.backend.main:app --host 0.0.0.0 --port 8000 > backend.log 2>&1 &

# Give the backend a moment to attempt to initialize.
echo "Waiting for backend to initialize (20 seconds)..."
sleep 20 

# --- Check if the Backend is Actually Running ---
# We use pgrep to find a process that matches the uvicorn command.
if ! pgrep -f "uvicorn v2_multilingual_api.backend.main:app"; then
    # If the process is NOT found, it means it crashed.
    echo "---"
    echo "--- !!! BACKEND FAILED TO START !!! ---"
    echo "--- Displaying contents of backend.log: ---"
    cat backend.log
    echo "--- End of backend.log ---"
    # We exit with an error code, which will cause the Hugging Face build to fail clearly.
    exit 1
else
    # If the process IS found, everything is okay.
    echo "Backend started successfully. Starting frontend..."
fi

# --- Launch the Frontend ---
# This command only runs if the check above was successful.
streamlit run v2_multilingual_api.frontend/app.py --server.port 7860 --server.address 0.0.0.0