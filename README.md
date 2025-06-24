# Intelligent Hazard Classification Platform

This is an end-to-end intelligent platform that classifies food incident reports into **hazard types** and **product categories** using deep learning models. It provides a simple web interface powered by FastAPI and enables efficient classification with pre-trained BERT-based models.

---

## Project Structure

```
toxin-detection-app/  
├── app/ # FastAPI backend (core logic, model loading, API routes)  
├── client/ # HTML/CSS/JS frontend (lightweight UI)  
├── model_ckpt/ # Model checkpoints (not included in repo – see below)  
├── requirements.txt # Python dependencies  
├── Dockerfile # Docker build configuration  
└── README.md # Project documentation
```

---

##  Features

-  Predicts **hazard type** and **product category** from textual incident reports
-  Uses BERT-based fine-tuned models for classification
-  Lightweight frontend for user interaction
-  Offers Dockerized deployment for hassle-free setup
-  Modular FastAPI backend

---

## Tech Stack

#### Machine Learning
- `transformers` (HuggingFace) – BERT models
- `torch` – PyTorch inference
- `scikit-learn`, `numpy`, `pandas` – Utility and preprocessing

#### Backend
- `FastAPI` – Web framework for API
- `Uvicorn` – ASGI server

#### Frontend
- `HTML`, `CSS`, `JavaScript` – Simple static frontend

#### Deployment
- Docker – Containerized environment

---

## Local Setup (Without Docker)

#### 1. Clone the repository

```bash
git clone https://github.com/Janesh-e/hazard-detection-app.git
cd logicloom
````

#### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

#### 4. Add the model checkpoints

Due to their large size, trained model checkpoints are not included in this repository.

However, we’ve provided Jupyter notebooks inside the `model_training/` folder which were used to train both models.  
You can use them to **re-train the models** and save the final weights locally.

After training, place the saved models inside the `model_ckpt/` directory as follows:

```bash
mkdir model_ckpt
# Save your model folders as:
# model_ckpt/hazard_model/
# model_ckpt/product_model/
````

> ⚠️ Make sure the model folder names match those used in `config.py` or update the paths accordingly.

#### 5. Run the app

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open `toxin_detection_dashboard.html` on a browser and send requests to the backend!

---

## Dockerized Setup (Recommended)

#### Option 1: Using Prebuilt Docker Image (if pushed)

```bash
docker pull janeshe/toxin-detection-app
docker run -p 8000:8000 janeshe/toxin-detection-app
```

#### Option 2: Build Docker Image Locally

```bash
# From the project root (with Dockerfile present)
docker build -t toxin-detection-app .
docker run -p 8000:8000 toxin-detection-app
```

---

## Model Details

This project uses two separate BERT-based classification models:

1. **Hazard Classification**
    - Number of Classes: 10
2. **Product Classification**
    - Number of Classes: 22

Both models are loaded using `AutoTokenizer` and `AutoModelForSequenceClassification`.

---
