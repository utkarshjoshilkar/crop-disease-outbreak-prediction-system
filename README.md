# 🌾 Crop Disease Outbreak Prediction System

[![Scale: Production-Ready](https://img.shields.io/badge/Scale-Production--Ready-brightgreen)](https://github.com/utkarshjoshilkar/crop-disease-outbreak-prediction-system)
[![AI: Agentic Multi-Modal](https://img.shields.io/badge/AI-Agentic%20Multi--Modal-blue)](https://ollama.ai)
[![Tech: FastAPI + Streamlit](https://img.shields.io/badge/Tech-FastAPI%20%2B%20Streamlit-orange)](https://fastapi.tiangolo.com)

An end-to-end, multi-modal diagnostic and forecasting system designed to empower farmers and agricultural experts. The system integrates **Vision-based AI**, **Real-time Weather Intelligence**, and **Machine Learning Outbreak Forecasting** to detect diseases and predict future risks with high precision.

---

## 🚀 Core Features

### 1. Multi-Modal Diagnostics
- **Vision Analysis**: Uses `llava:7b` to analyze crop images for visual symptoms.
- **Agentic Insights**: High-level reasoning using `llama3.1:8b` for severity assessment and recommendations.
- **Multilingual Support**: Provides advice in native languages (Hindi, Marathi, etc.) using `deep-translator`.

### 2. Sugarcane Red Rot Forecast (New)
- **Real-Time Monitoring**: Integrated dashboard for Sugarcane Red Rot risk.
- **XGBoost Engine**: Calibrated for high-accuracy outbreak prediction based on 7-day weather trends.
- **Biological Drivers**: Context-aware reasoning (Humidity streaks, wet spells, thermal windows).

### 3. Environment Intelligence
- **Weather Integration**: Live 11-feature weather data (Temperature, Humidity, Pressure, Wind, Precipitation, Solar Radiation) via Open-Meteo.
- **Location-Aware**: Mandatory GPS-based calibration for localized risk assessment.

---

## 🏗️ System Architecture

1.  **Backend Agent (FastAPI)**: Coordinates vision analysis, ML prediction, and database logging.
2.  **Risk Monitor (Streamlit)**: A dedicated live dashboard for specific disease forecasting (e.g., Sugarcane Red Rot).
3.  **Local AI Layer (Ollama)**: Offline-first LLM inference for data privacy and low-latency response.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Python 3.10+**
- **Ollama** ([Download here](https://ollama.com))

### 2. Local LLM Setup
Pull the required models to your local machine:
```bash
ollama pull llama3.1:8b
ollama pull llava:7b
```
Ensure Ollama is running (`ollama serve`).

### 3. Project Installation
```bash
# Clone the repository
git clone https://github.com/utkarshjoshilkar/crop-disease-outbreak-prediction-system.git
cd crop-disease-outbreak-prediction-system

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Initialize Database
python init_db.py
```

---

## 🚦 How to Run

> [!TIP]
> Starting the main backend will automatically launch the Streamlit Risk Monitor in the background.

1.  **Start the Main System**:
    ```bash
    python main.py
    ```
    *Alternatively, for development*: `uvicorn main:app --port 8000 --reload`

2.  **Access the Points of Interest**:
    -   **Main User Interface**: [http://localhost:8000](http://localhost:8000)
    -   **Risk Monitor Dashboard**: [http://localhost:8501](http://localhost:8501)
    -   **Interactive API Docs (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 Project Structure

-   `main.py`: Entry point; manages the FastAPI app and Streamlit subprocess.
-   `data/`: contains the **Sugarcane Risk Monitor** (`app.py`, `model_engine.py`).
-   `services/`:
    -   `ml_service.py`: Generalized disease risk prediction using XGBoost.
    -   `llm_service.py`: Interface for Ollama models (Llava and Llama).
    -   `weather_service.py`: Real-time API integration with Open-Meteo and NASA.
-   `db/`: SQLAlchemy models and SQLite storage.
-   `static/`: Modern Tailwind-based frontend assets.

---

## 📜 Requirements
The system requires standard data science and web libraries including:
- FastAPI, Streamlit, Uvicorn
- Scikit-learn, XGBoost, Pandas, Plotly
- SQLAlchemy, HTTPX, Deep-Translator

---

## ⚖️ License
This project is for educational and research purposes. Please verify model predictions with certified agricultural experts before making high-stakes farm decisions.
