# Crop Disease Outbreak Prediction System

AI-based crop disease outbreak prediction and detection system using weather data, machine learning, and agentic AI models.

## Features Included
- **Frontend Interface:** Built with HTML and Tailwind CSS (`static/index.html`).
- **Backend API:** Built with FastAPI (`main.py`).
- **Machine Learning Analysis:** Model predictions using XGBoost (`services/ml_service.py`).
# Crop Disease Outbreak Prediction System

AI-based crop disease outbreak prediction and detection system using weather data, machine learning, and agentic AI models.

## Features Included
- **Frontend Interface:** Built with HTML and Tailwind CSS (`static/index.html`).
- **Backend API:** Built with FastAPI (`main.py`).
- **Machine Learning Analysis:** Model predictions using XGBoost (`services/ml_service.py`).
- **Agentic Insights:** Integrated local LLM generation for multi-language farmer advice (`services/llm_service.py`).
- **Data Persistence:** SQLite database via SQLAlchemy (`db/models.py`).

## Prerequisites
- Python 3.8+ installed on your system.

## Requirements

The project relies on external libraries mapped in `requirements.txt`. Key dependencies include:
- `fastapi` and `uvicorn` for the backend application
- `xgboost`, `scikit-learn`, `pandas`, `numpy` for ML analysis
- `sqlalchemy` for database models
- `deep-translator` for text translation handling
- `python-multipart` to process form data and images

## Setup Instructions

### Option A: Using Antigravity IDE (Recommended for AI-Assisted Devs)
If you are running this project from within the **Antigravity IDE**, you can simply ask the agent to run the project for you.
1. Open the project folder in Antigravity.
2. In the chat prompt, type: `install requirements and start the application` or `run the code`.
3. The AI agent will handle creating the virtual environment (if needed), installing all libraries from `requirements.txt`, starting the server, and providing you with the local host URL.

### Option B: Manual Setup

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

## Running the Application

1. **Start the FastAPI Server:**
   You can start the server using Uvicorn. From the root directory, run:
   ```bash
   .\venv\Scripts\python.exe -m uvicorn main:app --reload
   ```
   *(Note: If you activated the virtual environment in your terminal, simply running `uvicorn main:app --reload` will also work.)*

2. **Access the Application:**
   Once the server is running, open your web browser and go to:
   - **Main UI:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
   - **API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
