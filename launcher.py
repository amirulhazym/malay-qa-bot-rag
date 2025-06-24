# Full, Final, and Corrected Code for: launcher.py

import streamlit as st
import subprocess
import time
import os

# --- Use Streamlit's session state to ensure this runs only ONCE ---
if 'backend_process' not in st.session_state:
    print("--- LAUNCHER: First run. Starting backend server... ---")
    
    # We construct the command to run uvicorn
    backend_command = [
        "uvicorn",
        "v2_multilingual_api.backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000"
    ]
    
    # Open log files for the backend's output and errors
    backend_log = open("backend.log", "w")
    
    # Start the backend process and store it in the session state
    st.session_state.backend_process = subprocess.Popen(
        backend_command, stdout=backend_log, stderr=backend_log
    )

    print(f"--- LAUNCHER: Backend process started with PID: {st.session_state.backend_process.pid} ---")
    print("--- LAUNCHER: Waiting 20 seconds for backend to initialize models... ---")
    time.sleep(20)

    # --- Health Check for the Backend ---
    # Check if the process crashed during startup
    if st.session_state.backend_process.poll() is not None:
        print("--- LAUNCHER: !!! BACKEND FAILED TO START ON INITIALIZATION !!! ---")
        backend_log.close()
        with open("backend.log", "r") as f:
            print(f.read())
        st.error("The backend server failed to start. Please check the logs.")
        # We use st.stop() to halt the app if the backend fails
        st.stop()
    else:
        print("--- LAUNCHER: Backend health check passed. Process is running. ---")
        
# --- Step 3: Launch the Streamlit Frontend ---
# This part of the code will run every time, but the backend is already running.
# The command is to run the *actual* frontend app file.

print("--- LAUNCHER: Handing off to the main Streamlit frontend application. ---")

# We use os.system for simplicity to run the final command
frontend_command = (
    "streamlit run v2_multilingual_api/frontend/app.py "
    "--server.port 7860 "
    "--server.address 0.0.0.0"
)

os.system(frontend_command)