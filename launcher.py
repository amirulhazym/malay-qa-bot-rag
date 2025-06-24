# Full Code for: launcher.py

import subprocess
import time
import os

# --- Step 1: Launch the FastAPI Backend ---
# We use subprocess.Popen to start the backend in a new process,
# which runs in the background. We redirect its output to a log file.

print("--- LAUNCHER: Starting backend server... ---")

# We construct the command to run uvicorn
backend_command = [
    "uvicorn",
    "v2_multilingual_api.backend.main:app",
    "--host", "0.0.0.0",
    "--port", "8000"
]

# Open log files for the backend's output and errors
backend_log = open("backend.log", "w")

# Start the backend process
backend_process = subprocess.Popen(backend_command, stdout=backend_log, stderr=backend_log)

print(f"--- LAUNCHER: Backend process started with PID: {backend_process.pid} ---")
print("--- LAUNCHER: Waiting 20 seconds for backend to initialize models... ---")
time.sleep(20)

# --- Step 2: Health Check for the Backend ---
# Check if the process is still running. If poll() returns a number, it means it crashed.
if backend_process.poll() is not None:
    print("--- LAUNCHER: !!! BACKEND FAILED TO START !!! ---")
    # Read the log file to find out why it failed
    backend_log.close()
    with open("backend.log", "r") as f:
        print(f.read())
    # Exit the launcher with an error code to make the build fail clearly
    exit(1)
else:
    print("--- LAUNCHER: Backend health check passed. Process is running. ---")


# --- Step 3: Launch the Streamlit Frontend ---
# This command will run in the foreground and take over the main process.
# This is what the Hugging Face Spaces platform expects.

print("--- LAUNCHER: Starting frontend server... ---")

# We use os.system for simplicity to run the final command
frontend_command = (
    "streamlit run v2_multilingual_api/frontend/app.py "
    "--server.port 7860 "
    "--server.address 0.0.0.0"
)

os.system(frontend_command)

# Close the log file when the app is shut down
backend_log.close()