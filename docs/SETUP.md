# Local Setup and Deployment Guide

This guide details how to install, configure, test, and run **NutriFresh AI** across Windows, macOS, Linux, and Docker environments.

---

## 💻 Prerequisites
- Python 3.10 or 3.11 installed
- Git installed
- (Optional) Docker & Docker Compose

---

## 🚀 1. Local Python Setup

### Step 1: Clone Repository
```bash
git clone https://github.com/<your-username>/nutrifresh-ai.git
cd nutrifresh-ai
```

### Step 2: Create & Activate Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **macOS / Linux:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
# Windows:
copy .env.example .env

# Linux / macOS:
cp .env.example .env
```

---

## 🧪 2. Run Tests
Verify system integrity by running the test suite:
```bash
pytest tests/ -v
```

---

## ⚡ 3. Running the Application

### Option A: Start FastAPI Web Server & UI
```bash
# Using CLI tool:
python -m src.cli.main serve --port 8000 --reload

# Or directly with Uvicorn:
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
- Open Interactive Dashboard: **`http://localhost:8000/app`**
- Open Swagger API Documentation: **`http://localhost:8000/docs`**

---

### Option B: Using the CLI Tool
```bash
# Single image prediction
python -m src.cli.main predict --image samples/fresh_apple.jpg

# Single image with JSON output
python -m src.cli.main predict --image samples/rotten_apple.webp --json

# Batch evaluate an entire folder
python -m src.cli.main evaluate --dir samples/ --output batch_results.json
```

---

## 🐳 4. Docker Deployment

### Run with Docker Compose:
```bash
docker compose up --build
```

### Run standalone Docker container:
```bash
docker build -t nutrifresh-ai:latest .
docker run -p 8000:8000 nutrifresh-ai:latest
```

The containerized service will be available at `http://localhost:8000/app`.
