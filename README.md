# Medical AI Prediction Assistant — One-Day Demo

Educational machine-learning demo using synthetic data. It is not a clinical tool.

## Requirements
- Python 3.10 or newer
- Internet connection for installing packages

## Windows setup
Open a terminal in this folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed in the terminal, usually http://localhost:8501.

## What it demonstrates
- Synthetic dataset generation
- Train/test split
- Random Forest classification
- Basic held-out evaluation
- Interactive local web UI

The measurements, labels, and scores are invented for demonstrating a machine-learning workflow. Do not use this app for health decisions.
