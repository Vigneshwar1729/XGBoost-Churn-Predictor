# 🏦 Predictive Analytics in Banking: XGBoost vs FT-Transformer

![Python](https://img.shields.io/badge/Python-3.11-blue.svg) ![FastAPI](https://img.shields.io/badge/FastAPI-0.110.0-009688.svg) ![React](https://img.shields.io/badge/React-18-61DAFB.svg) ![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-F37626.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.14.0-EE4C2C.svg) ![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)

---

## 1. Executive Summary
Customer attrition, commonly known as customer churn, is one of the most critical challenges faced by financial institutions today. In the highly competitive banking sector, the cost of acquiring a new customer is significantly higher than retaining an existing one. This repository presents a comprehensive, end-to-end Machine Learning and Deep Learning pipeline designed to accurately predict whether a bank customer is at a high risk of churning.

To achieve maximum accuracy and derive actionable business intelligence, this project implements, evaluates, and deploys two distinct predictive paradigms side-by-side:
1. **XGBoost:** A highly optimized, traditional tree-based ensemble method.
2. **FT-Transformer:** A state-of-the-art Deep Learning architecture specifically designed for tabular data using the PyTorch Tabular framework.

The final deliverable is a production-ready, containerized full-stack web application (React + FastAPI) deployed to the cloud, providing non-technical stakeholders with a seamless interface to evaluate customer risk in real-time.

---

## 2. Introduction & Problem Statement
### 2.1 The Business Context
Banks invest heavily in marketing to attract customers. When a customer leaves, the bank loses not only the immediate revenue but also the lifetime value of that customer. If a bank can accurately predict *which* customers are likely to leave, their retention and customer success teams can intervene proactively with targeted promotions.

### 2.2 Challenges Solved
* **Imbalanced Data:** Churn datasets are inherently imbalanced (e.g., 80% of customers stay, 20% leave). A naive model could simply guess "Stay" every time and achieve 80% accuracy, providing zero business value.
* **Algorithmic Selection:** While traditional Machine Learning has historically dominated tabular data, modern Deep Learning (Transformers) is beginning to show superiority. This project validates this hypothesis.

---

## 3. Data Description & Preprocessing
### 3.1 The Dataset
The models were trained on an anonymized dataset comprising 10,000 bank customers, featuring:
* **Demographics:** `Age`, `Gender`, `Geography`
* **Financial Standing:** `CreditScore`, `Balance`, `EstimatedSalary`
* **Behavioral Metrics:** `Tenure`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`
* **Target Variable:** `Exited` (1 = Churned, 0 = Stayed).

### 3.2 Feature Engineering
* **Train/Validation/Test Split:** The dataset was rigorously partitioned to prevent data leakage.
* **Scaling:** Continuous variables were normalized to ensure neural networks converge quickly.
* **Encoding:** Categorical variables were transformed into numerical formats for XGBoost and natively tokenized for the FT-Transformer.

---

## 4. Methodology 1: Traditional Machine Learning (XGBoost)
### 4.1 Theory & Implementation
eXtreme Gradient Boosting (XGBoost) builds a series of decision trees sequentially, where each new tree specifically attempts to correct the errors (residuals) made by the previous trees.

### 4.2 Handling Class Imbalance & Tuning
To combat the 80/20 imbalanced nature of the dataset, we utilized the `scale_pos_weight` parameter, forcing the algorithm to heavily penalize errors made on churners. We utilized `GridSearchCV` with 3-fold cross-validation to search for optimal hyperparameters (`n_estimators`, `max_depth`, `learning_rate`), optimizing specifically for the **F1-Score**.

---

## 5. Methodology 2: Deep Learning (FT-Transformer)
### 5.1 The Tabular Data Challenge
While Neural Networks dominate image processing, they historically struggle on structured tabular data. The **Feature Tokenizer + Transformer (FT-Transformer)** solves this:
1. **Feature Tokenizer:** Converts all inputs (categorical and continuous) into dense vector embeddings.
2. **Transformer (Self-Attention):** Passes tokens through multi-head self-attention blocks, learning complex, non-linear relationships (e.g., how `Age` uniquely interacts with `Balance` depending on `Geography`).

### 5.2 PyTorch Tabular Implementation
Configured with 3 Attention Blocks, 4 Attention Heads, and a 128-64-32 Feed-Forward layer, the model was trained using an advanced AdamW optimizer over 20 epochs.

---

## 6. System Architecture & Full-Stack Engineering

### 6.1 The FastAPI Backend
FastAPI was chosen for its extreme speed and native data validation via `Pydantic`. Upon server startup, both the XGBoost `.json` model and the PyTorch FT-Transformer weights are loaded into RAM. The `/predict` endpoint processes incoming data through both models concurrently, returning a nested comparative payload.

### 6.2 The React + Vite Frontend
A modern Single Page Application (SPA) was built using React and TypeScript. It features a "glassmorphism" aesthetic and dynamically parses the dual-payload response to instantly render side-by-side comparison cards.

### 6.3 Docker & Cloud Deployment (Render)
* **The PyTorch Memory Optimization:** Deploying PyTorch typically downloads 3GB+ of CUDA GPU drivers, instantly crashing free-tier cloud servers (OOM error). We engineered a Docker bypass (`pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu`) to strictly download lightweight CPU wheels, shrinking the image by 80%.
* **Zero Latency Cron Job:** To prevent the cloud server from "sleeping" after 15 minutes of inactivity, a `/health` endpoint is continually pinged by a chronometer every 14 minutes, mathematically guaranteeing zero-latency predictions 24/7.

---

## 7. Results & Business Conclusion

Because churn is highly imbalanced, we prioritized **F1-Score** and **ROC-AUC** over simple Accuracy.

### 7.1 XGBoost Performance
* **Accuracy:** 74.0%
* **ROC-AUC Score:** 85.3%
* **Recall:** 72.2%
* **Precision:** 48.9%
* **F1-Score:** 58.3%

### 7.2 FT-Transformer Performance
* **Accuracy:** 86.1%
* **ROC-AUC Score:** 90.2%
* **Recall:** 75.5%
* **Precision:** 62.7%
* **F1-Score:** 68.4%

**Key Insight:** While XGBoost achieved a strong Recall, it suffered from a high rate of False Positives (low Precision), which would cause the bank to waste marketing budgets on loyal customers. The **FT-Transformer vastly outperformed XGBoost**, leveraging deep attention mechanisms to increase both Recall and Precision. By deploying the FT-Transformer, the bank can confidently execute highly targeted retention campaigns, maximizing ROI.

---

## 💻 Running the Project Locally

### 1. Setup & Train the Models
```bash
# Install dependencies
pip install -r requirements.txt

# Split the dataset into Train, Validation, and Test sets
python split_data.py --input Customer-Churn-Records.csv --output_dir ./splits

# Train XGBoost
python train_xgboost.py

# Train the FT-Transformer (PyTorch Deep Learning)
python train_ft_transformer.py
```

### 2. Start the FastAPI Backend
```bash
uvicorn app:app --reload
```
*API Documentation is automatically generated at `http://localhost:8000/docs`.*

### 3. Start the React Frontend
```bash
cd frontend
npm run dev
```
*Access the beautiful UI at `http://localhost:5173`.*
