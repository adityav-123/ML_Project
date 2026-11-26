# CareLens — Disease Prediction System

CareLens is an educational machine-learning web application that analyzes selected symptoms and returns a possible disease pattern, confidence estimate, and alternative patterns.

The application uses a FastAPI backend served with Uvicorn and a responsive HTML/CSS/JavaScript dashboard. It is not a medical diagnostic tool and should not replace advice from a qualified healthcare professional.

## Features

- Symptom search and multi-select workflow
- Minimum evidence requirement before prediction
- Disease prediction with confidence and alternatives
- Random Forest model evaluation
- Probability-based evidence scoring for user predictions
- Dataset and model performance dashboard
- FastAPI REST endpoints
- Responsive dashboard UI

## Project structure

```text
Disease_prediction/
├── app.py
├── disease_prediction.py
├── requirements.txt
├── Training(disease).csv
├── Testing(disease).csv
└── static/
    ├── index.html
    ├── style.css
    └── app.js
```

## Requirements

- Python 3.10 or newer
- Git
- A modern web browser

## Windows installation

Open PowerShell in the project folder:

```powershell
git clone https://github.com/adityav-123/ML_Project.git
cd ML_Project

py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

## Windows run command

```powershell
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

To stop the server, press `Ctrl+C`.

## macOS installation

Open Terminal in the project folder:

```bash
git clone https://github.com/adityav-123/ML_Project.git
cd ML_Project

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## macOS run command

```bash
python -m uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser.

To stop the server, press `Ctrl+C`.

## API endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/health` | Check whether the API and model are ready |
| GET | `/api/overview` | Retrieve dataset and model metrics |
| GET | `/api/symptoms` | Retrieve available symptoms |
| POST | `/api/predict` | Generate a prediction from selected symptoms |

Example prediction request:

```json
{
  "symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"]
}
```

## Optional command-line model check

After activating the virtual environment:

Windows:

```powershell
python disease_prediction.py
```

macOS:

```bash
python3 disease_prediction.py
```

## Dataset

The current dataset contains symptom columns and a `prognosis` disease label. The application removes unnecessary unnamed columns, cleans duplicate records, converts symptom values to numeric values, and encodes disease names before training.

## Disclaimer

This project is intended for educational and academic purposes only. Predictions are generated from a machine-learning dataset and must not be treated as medical diagnosis, treatment advice, or emergency guidance. Always consult a qualified healthcare professional.
