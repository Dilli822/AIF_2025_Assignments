# Disease Predictor

This project uses a trained Machine Learning model to predict diseases based on symptoms provided by the user through an API.

---

## Features

* Predict disease from common symptoms using a trained ML model.
* FastAPI-based RESTful API.
* Dockerized for easy deployment.
* Supports environment-based configuration using `pydantic`.
* Includes unit testing with `pytest`.
* Linting and formatting with pre-commit hooks.
* MkDocs for documentation.

---

## 📁 Project Structure

```bash
disease-predictor/
├── app/                     # Application code
│   ├── main.py              # API entry point
│   ├── ml_model.py          # Model loading and prediction
│   ├── schemas.py           # Request/response schemas
│   ├── crud.py              # CRUD operations (if applicable)
│   ├── database.py          # DB connection (optional)
│   └── logging-config.py    # Logging setup
├── data/                    # Input CSV or DB
│   └── disease_predict.csv
├── models/                  # Trained ML models
│   └── disease_model.joblib
├── scripts/
│   └── train_model.py       # Model training script
├── tests/
│   ├── test.py
│   └── test_predict.py
├── .env                     # Environment variables
├── .gitignore
├── .pre-commit-config.yaml  # Pre-commit config
├── Dockerfile
├── docker-compose.yml       # Optional services
├── mkdocs.yml               # MkDocs configuration
└── README.md
```

---

## 🔧 Requirements

### For Local Setup:

* Python 3.8+
* pip
* virtualenv (recommended)

### For Docker Setup:

* Docker
* Docker Compose (optional, for multi-service setup)

---

## ✅ Setup Instructions

### 🖥️ Run Locally

1. **Clone the repository**

```bash
git clone https://github.com/dilli822/AIF-2025.git # IT IS PRIVATE REPO
cd disease-predictor
```

2. **Create virtual environment**

```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install dependencies**

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. **Train the model (if needed)**

```bash
 python scripts/train_model.py. # BUT THE MODEL SAVED VERSION IS AVAILABLE
```

5. **Run the API**

```bash
uvicorn app.main:app --reload

if occupied use uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 || uvicorn app.main:app --reload
```

6. **Access the API documentation**

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🐳 Run with Docker

1. **Build Docker image**

```bash
docker build -t diseasepredict .
docker decompose part:
docker-compose up --build
```

2. **Run the container**

```bash
docker run -d -p 9000:9000 --env-file .env diseasepredictor
```

3. **Check API in browser**

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🧪 Run Tests

```bash
pytest
```

---

### 🧹 Pre-commit Hooks (optional)

1. **Install pre-commit**

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files

```

---

### 📄 Sample API Request

**Endpoint:** `POST /predict`

**Request Body:**

```json
{
  "fever": true,
  "cough": true,
  "skin_rash": false,
  "joint_pain": false,
  "stomach_pain": true,
  "vomiting": false,
  "fatigue": true,
  "headache": false
}
```

**Response Example:**

```json
{
  "predicted_disease": "Typhoid"
}
```

---

## Documentation

To build docs locally (optional):

```bash
if 8001 is occupied: use other port
mkdocs serve --dev-addr=0.0.0.0:8001
```

Visit: [http://127.0.0.1:8000](http://127.0.0.1:8000/)

---
