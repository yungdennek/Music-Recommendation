# Music-Recommendation
Finding music recommendations based on people's stated music taste using Machine Learning methods

## Local Development Setup

- **1. Create and activate virtual environment**
  - macOS / Linux:
    - `python3 -m venv .venv`
    - `source .venv/bin/activate`
  - Windows (PowerShell):
    - `python -m venv .venv`
    - `.venv\Scripts\Activate.ps1`

- **2. Install dependencies**
  - `pip install -r requirements.txt`

- **3. Run tests (TDD workflow)**
  - Run all tests: `pytest`
  - Run specific test file: `pytest tests/test_data_loader.py`

Project code lives under `src/`, and tests under `tests/`.

## Docker Setup

This project also supports a containerized workflow so you do not need to manage Python dependencies locally.

- **1. Build image**
  - `docker build -t music-recommendation .`

- **2. Run tests inside container**
  - `docker run --rm music-recommendation pytest`

You can also start an interactive shell inside the container:

- `docker run --rm -it music-recommendation /bin/bash`
