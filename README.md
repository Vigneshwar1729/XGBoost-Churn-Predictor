# 🏦 XGBoost and FT-Transformer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.11-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg) ![React](https://img.shields.io/badge/React-18-61DAFB.svg) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-F37626.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-EE4C2C.svg) ![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)

## 📌 About This Project
Customer churn (when clients stop doing business with a company) is one of the most critical metrics for modern banks. Retaining an existing customer is significantly cheaper than acquiring a new one. 

This repository contains an **end-to-end Machine Learning and Deep Learning pipeline** designed to predict whether a bank customer is at high risk of churning. It serves as a comprehensive Predictive Analytics project, directly contrasting a highly optimized traditional Tree-based approach (**XGBoost**) against a state-of-the-art Deep Learning architecture specifically designed for tabular data (**FT-Transformer**).

**The Business Value:** By feeding customer metrics (such as age, tenure, balance, and number of products) into this model, the bank's retention team can proactively identify high-risk individuals and offer targeted promotions *before* they leave, saving significant revenue.

### The Dataset
The models were trained on an anonymized dataset of 10,000 bank customers. After extensive feature selection to prevent data leakage, the following behavioral and demographic features were utilized:
`Credit Score`, `Gender`, `Age`, `Tenure`, `Balance`, `Number of Products`, `Credit Card Status`, and `Active Member Status`.

---

## 🚀 Architecture Overview

1. **The Brains (Machine Learning & Deep Learning):** 
   * **XGBoost:** A traditional gradient boosting machine, tuned via `GridSearchCV` to handle imbalanced data.
   * **FT-Transformer:** A modern deep learning architecture powered by `pytorch_tabular`, leveraging attention mechanisms to interpret tabular customer data.
2. **The API (Backend):** A highly concurrent **FastAPI** web server that loads both trained models into memory and exposes a single `/predict` endpoint to process and return comparative predictions.
3. **The Face (Frontend):** A premium **React + TypeScript + Vite** user interface featuring a side-by-side comparative layout, making the dual-model predictions easily accessible to non-technical stakeholders.

---

## 📊 Model Performance & Metrics

By comparing these two paradigms side-by-side on the hold-out dataset, we demonstrate the performance evolution on tabular data. Because this is an imbalanced dataset (fewer people churn than stay), **F1-Score and ROC-AUC** are heavily prioritized alongside Accuracy.

### XGBoost (Traditional ML)
* **Accuracy:** `74.0%`
* **ROC-AUC Score:** `85.3%`
* **Recall:** `72.2%`
* **Precision:** `48.9%`
* **F1-Score:** `58.3%`

### FT-Transformer (Deep Learning)
* **Accuracy:** `86.1%`
* **ROC-AUC Score:** `90.2%`
* **Recall:** `75.5%`
* **Precision:** `62.7%`
* **F1-Score:** `68.4%`

**Key Insight:** While XGBoost aggressively caught potential churners (high recall), it suffered from false positives. The FT-Transformer architecture maintained high recall while drastically improving precision, leading to a much stronger F1-Score and making it the vastly superior model for targeted business interventions.

---

## 💻 Running the Project Locally

### 1. Setup & Train the XGBoost Model
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
*(Note: If you run XGBoost, it will finish in seconds. If you run FT-Transformer without a GPU, it will take considerably longer.)*

### 2. Train the FT-Transformer Model (Deep Learning)
Deep Learning models require PyTorch to train. To build the FT-Transformer:
```bash
# Install PyTorch Tabular
pip install torch pytorch-tabular

# Train the architecture (will save to ./ft_transformer_model/)
python train_ft_transformer.py
```

### 3. Start the FastAPI Backend
Open a terminal and run the local server. It will automatically load both the `xgboost_churn_model.json` and the `ft_transformer_model`.
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
3. Select **Docker** as the runtime. Render will automatically detect the `Dockerfile`, install the lightweight CPU version of PyTorch, load both models, and build the API container.

### Frontend UI (Static Site)
1. Create a new "Static Site" on Render.
2. Connect this same GitHub repository.
3. **Build Command:** `cd frontend && npm install && npm run build`
4. **Publish Directory:** `frontend/dist`
5. **Environment Variables:** Add `VITE_API_URL` and set it to the URL of your deployed Backend Web Service (e.g., `https://your-backend.onrender.com`).
