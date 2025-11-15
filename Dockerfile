# --- 1. Base Image: The "Box" ---
# We'll use an official, slim version of Python 3.11 as our starting point.
FROM python:3.11-slim

# --- 2. Set Working Directory: The "Floor" ---
# Create a folder inside the container called '/app' and move into it.
WORKDIR /app

# --- 3. Install Dependencies: The "Tools" ---
# Copy *only* the requirements file first.
COPY requirements.txt .

# Run pip install.
# This uses Docker's cache: if requirements.txt doesn't change,
# this step won't re-run, making future builds much faster.
RUN pip install --no-cache-dir -r requirements.txt

# --- 4. Copy Project Code: The "Payload" ---
# Copy *everything* else from your local folder into the container's '/app' folder.
# This includes:
# - main.py
# - tools.py
# - build_rag.py
# - data/ (CRITICAL: This copies your .faiss files)
COPY . .

# --- 5. Expose Port: The "Label" ---
# FIX: Change this from 8000 to 8080
EXPOSE 8080

# --- 6. Run Command: The "On Switch" ---
#
# CRITICAL: We use '--host 0.0.0.0' (not '127.0.0.1')
# FIX: We change the port from 8000 to 8080 to match Cloud Run
CMD ["/bin/sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]