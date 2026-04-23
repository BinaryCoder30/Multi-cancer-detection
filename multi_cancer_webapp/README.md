# Multi-Cancer Diagnostic Web App

This project provides a complete end-to-end solution for diagnosing both **Colon** and **Lung** cancer from histopathological images. It uses a **FastAPI backend** to serve two PyTorch models simultaneously, and a **Streamlit frontend** for a seamless user experience.

## Features
- **Dual Model Support:** Runs both Colon and Lung models.
- **Cross-Model Validation:** Intelligent heuristic that double checks images. If you upload a Lung image but select the Colon category, the system will detect the mismatch and warn you.
- **Premium UI:** Dynamic and modern UI built with Streamlit.

## Setup Instructions

### 1. Start the Backend
Open a terminal, navigate to the `backend` directory:
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload
```

### 2. Start the Frontend
Open a **new** terminal, navigate to the `frontend` directory:
```bash
cd frontend
pip install -r requirements.txt
python -m streamlit run app.py
```

### 3. Usage
Navigate to the Streamlit UI in your browser (usually `http://localhost:8501`).
Select the tissue type (Colon or Lung), upload your image, and generate predictions!
