# Use the official Python 3.10 image as our base
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /code

# Install system-level dependencies required by our app and Git LFS
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make our startup script executable
RUN chmod +x ./setup.sh

# Expose the port the Streamlit app will run on
EXPOSE 7860

# The command that will be run when the container starts
CMD ["bash", "./setup.sh"]