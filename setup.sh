#!/bin/bash

# This line tells the server that this is a shell script

# Launch the FastAPI backend API in the background.
# The '&' symbol at the end is crucial. It tells the server to run this command
# and immediately move on to the next one without waiting for it to finish.
echo "Starting backend server..."
uvicorn v2_multilingual_api.backend.main:app --host 0.0.0.0 --port 8000 &

# Give the backend a moment to initialize the AI models.
# Loading models can take time. This 'sleep' command pauses for 20 seconds
# to ensure the backend is ready before the frontend starts trying to talk to it.
echo "Waiting for backend to initialize..."
sleep 20 

# Launch the Streamlit frontend.
# This is the final command, so it runs in the foreground.
# --server.port 7860 is the required port for Streamlit apps on Hugging Face.
# --server.address 0.0.0.0 makes it accessible.
echo "Starting frontend server..."
streamlit run v2_multilingual_api.frontend/app.py --server.port 7860 --server.address 0.0.0.0
