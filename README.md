# 🏦 Bank Customer Churn Prediction: Predictive Analytics for Business

![Python](https://img.shields.io/badge/Python-3.11-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.103.0-009688.svg) ![React](https://img.shields.io/badge/React-18.2.0-61DAFB.svg) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-F37626.svg) ![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)

## 📌 About This Project
Customer churn (when clients stop doing business with a company) is one of the most critical metrics for modern banks. Retaining an existing customer is significantly cheaper than acquiring a new one. 

This repository contains an **end-to-end Machine Learning pipeline and web application** designed to predict whether a bank customer is at high risk of churning. It serves as a comprehensive Predictive Analytics internship project, contrasting a traditional Tree-based approach (XGBoost) against a Deep Learning architecture (FT-Transformer).

**The Business Value:** By feeding customer metrics (such as age, tenure, balance, and number of products) into this model, the bank's retention team can proactively identify high-risk individuals and offer targeted promotions *before* they leave, saving significant revenue.

### The Dataset
The model was trained on an anonymized dataset of 10,000 bank customers. After extensive feature selection to prevent data leakage, the following behavioral and demographic features were utilized:
`Credit Score`, `Gender`, `Age`, `Tenure`, `Balance`, `Number of Products`, `Credit Card Status`, and `Active Member Status`.

---

## 🚀 Architecture Overview

1. **The Brain (Machine Learning):** An **XGBoost Classifier** optimized via `GridSearchCV` to handle highly imbalanced data (using `scale_pos_weight`). 
2. **The API (Backend):** A highly concurrent **FastAPI** web server that loads the trained `.json` model into memory and exposes a `/predict` endpoint.
3. **The Face (Frontend):** A minimal, premium **React + TypeScript + Vite** user interface featuring glassmorphism aesthetics to make predictions accessible to non-technical stakeholders.

---

## 📊 Model Performance & Metrics

The XGBoost model was mathematically tuned across 81 different hyperparameter combinations using 3-fold cross-validation. The final validation metrics on the hold-out dataset are:

* **Accuracy:** `78.9%`
* **ROC-AUC Score:** `85.3%` *(Excellent distinction between classes)*
* **Recall:** `72.2%` *(Successfully catches the vast majority of actual churners)*
* **Precision:** `48.9%`
* **F1-Score:** `58.3%`

**Key Insight:** The `NumOfProducts` feature was determined to be the most critical indicator of customer loyalty, accounting for **34.5%** of the model's decision-making weight.

---

## 💻 Running the Project Locally

### 1. Setup & Train the Model
```bash
# Install dependencies
pip install -r requirements.txt

# Split the dataset into Train, Validation, and Test sets
python split_data.py --input Customer-Churn-Records.csv --output_dir ./splits

# Train the model and log to Weights & Biases
python train_xgboost.py

# Evaluate the final model on the validation set
python evaluate_xgboost.py
```

### 2. Start the FastAPI Backend
Open a terminal and run the local server. It will automatically load `xgboost_churn_model.json`.
```bash
uvicorn app:app --reload
```
*API Documentation is automatically generated at `http://localhost:8000/docs`.*

### 3. Start the React Frontend
Open a **new terminal** (leave the backend running) and start the UI:
```bash
cd frontend
npm run dev
```
*Access the beautiful UI at `http://localhost:5173`.*

---

## ☁️ Cloud Deployment (Render)

This project is fully containerized and configured for free, continuous deployment on [Render](https://render.com/).

### Backend API (Web Service)
1. Create a new "Web Service" on Render.
2. Connect this GitHub repository.
3. Select **Docker** as the runtime. Render will automatically detect the `Dockerfile` and build the API container.

### Frontend UI (Static Site)
1. Create a new "Static Site" on Render.
2. Connect this same GitHub repository.
3. **Build Command:** `cd frontend && npm install && npm run build`
4. **Publish Directory:** `frontend/dist`
5. **Environment Variables:** Add `VITE_API_URL` and set it to the URL of your deployed Backend Web Service (e.g., `https://your-backend.onrender.com`).
