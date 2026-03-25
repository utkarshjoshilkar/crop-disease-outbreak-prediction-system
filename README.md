# Crop Disease Outbreak Prediction System

AI-based crop disease outbreak prediction and detection system using weather data, machine learning, and agentic AI models.

## Features Included
- **Frontend Interface:** Built with HTML and Tailwind CSS (`static/index.html`).
- **Backend API:** Built with FastAPI (`main.py`).
- **Machine Learning Analysis:** Model predictions using XGBoost (`services/ml_service.py`).
- **Agentic Insights:** Integrated local LLM generation for multi-language farmer advice (`services/llm_service.py`).
- **Data Persistence:** SQLite database via SQLAlchemy (`db/models.py`).

## Prerequisites
- **Python 3.8+** installed on your system.
- **Ollama** installed on your system.

## Setup Instructions

### 1. Install Ollama
Ollama is required to run the local LLM models (`llama3.1:8b` and `llava:7b`).
1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Once installed, pull the required models by running:
   ```bash
   ollama pull llama3.1:8b
   ollama pull llava:7b
   ```
3. Ensure Ollama is running in the background.

### 2. Manual Project Setup

1. **Create and Activate a Virtual Environment:**
   Run the following commands in the project root directory:
   ```bash
   python -m venv venv
   
   # For Windows:
   .\venv\Scripts\activate
   
   # For macOS/Linux:
   source venv/bin/activate
   ```

2. **Install Required Dependencies:**
   Ensure the virtual environment is activated, then install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Database:**
   Run the initialization script to set up the SQLite database:
   ```bash
   python init_db.py
   ```

## Running the Application

1. **Start the FastAPI Server:**
   You can start the server using Uvicorn. From the root directory, run:
   ```bash
   uvicorn main:app --reload
   ```

2. **Access the Application:**
   Once the server is running, open your web browser and go to:
   - **Main UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - **API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## Project Structure
- `main.py`: Entry point for the FastAPI application.
- `services/`: Contains business logic (ML service, LLM service, Weather service).
- `db/`: Database models and connection management.
- `static/`: Frontend assets (HTML, JS, CSS).
- `config.py`: Configuration settings for LLM and APIs.

##llm model setup on local pc

1. Download and install Ollama from [ollama.com](https://ollama.com).
2. Once installed, pull the required models by running:
   ```bash
   ollama pull llama3.1:8b
   ollama pull llava:7b
   ```

3. Ensure Ollama is running in the background.
   ```bash
   ollama serve
   ```

4. Verify the models are loaded by running:
   ```bash
   ollama list
   ```

