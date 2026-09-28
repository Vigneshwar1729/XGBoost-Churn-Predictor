# Comprehensive Project Report: AI-Powered Bank Customer Churn Prediction
## A Comparative Study of XGBoost and Deep Learning (FT-Transformer)

---

## 1. Executive Summary
Customer attrition, commonly known as customer churn, is one of the most critical challenges faced by financial institutions today. In the highly competitive banking sector, the cost of acquiring a new customer is widely recognized to be significantly higher than retaining an existing one. This project presents a comprehensive, end-to-end Machine Learning and Deep Learning pipeline designed to accurately predict whether a bank customer is at a high risk of churning.

To achieve maximum accuracy and derive actionable business intelligence, this project implements, evaluates, and deploys two distinct predictive paradigms side-by-side:
1. **XGBoost:** A highly optimized, traditional tree-based ensemble method.
2. **FT-Transformer:** A state-of-the-art Deep Learning architecture specifically designed for tabular data using the PyTorch Tabular framework.

The final deliverable is a production-ready, containerized full-stack web application (React + FastAPI) deployed to the cloud, providing non-technical stakeholders with a seamless interface to evaluate customer risk in real-time.

---

## 2. Introduction & Problem Statement
### 2.1 The Business Context
Banks invest heavily in marketing and onboarding procedures to attract customers. When a customer leaves (closes their accounts, moves assets to competitors, etc.), the bank loses not only the immediate revenue but also the lifetime value of that customer. If a bank can accurately predict *which* customers are likely to leave, their retention and customer success teams can intervene proactively with targeted promotions, waived fees, or personalized financial advice.

### 2.2 Objective
The primary objective of this project is to develop a predictive engine that ingests demographic and behavioral data of a bank's clientele and outputs a real-time risk assessment. 

### 2.3 Challenges
* **Imbalanced Data:** Churn datasets are inherently imbalanced (e.g., 80% of customers stay, 20% leave). A naive model could simply guess "Stay" every time and achieve 80% accuracy, but it would have zero business value.
* **Algorithmic Selection:** While traditional Machine Learning (like Random Forests and XGBoost) has historically dominated tabular data, modern Deep Learning (Transformers) is beginning to show superiority. This project aims to test this hypothesis.

---

## 3. Data Description & Preprocessing
### 3.1 The Dataset
The models were trained on an anonymized dataset comprising 10,000 bank customers. The dataset includes a mix of continuous and categorical variables:
* **Demographics:** `Age`, `Gender`, `Geography` (Country)
* **Financial Standing:** `CreditScore`, `Balance`, `EstimatedSalary`
* **Behavioral Metrics:** `Tenure` (years with the bank), `NumOfProducts` (number of bank products used), `HasCrCard` (has a credit card), `IsActiveMember` (frequent interaction with the bank).
* **Target Variable:** `Exited` (1 = Churned, 0 = Stayed).

### 3.2 Feature Engineering and Selection
To ensure the model learns generalized patterns rather than memorizing noise (which leads to overfitting), strict data preprocessing was applied:
* **Train/Validation/Test Split:** The dataset was rigorously partitioned. The model was trained on the training set, hyperparameter-tuned on the validation set, and final metrics were derived from the unseen test set to simulate real-world deployment.
* **Scaling:** Continuous variables (`Balance`, `EstimatedSalary`, etc.) were normalized to ensure algorithms like Neural Networks converge quickly.
* **Encoding:** Categorical variables (`Gender`, `Geography`) were transformed into numerical formats readable by the algorithms.

---

## 4. Methodology 1: Traditional Machine Learning (XGBoost)
### 4.1 Theoretical Background
eXtreme Gradient Boosting (XGBoost) is an advanced implementation of gradient boosting algorithms. It works by building a series of decision trees sequentially, where each new tree specifically attempts to correct the errors (residuals) made by the previous trees.

### 4.2 Handling Class Imbalance
To combat the 80/20 imbalanced nature of the dataset, we utilized the `scale_pos_weight` parameter. By calculating the ratio of negative class samples to positive class samples, we forced the XGBoost algorithm to heavily penalize errors made on the minority class (churners).

### 4.3 Hyperparameter Tuning
We utilized `GridSearchCV` combined with 3-fold cross-validation to exhaustively search for the optimal model architecture. Parameters tuned included:
* `n_estimators`: The number of sequential trees built.
* `max_depth`: The maximum depth of each tree (to control overfitting).
* `learning_rate`: The step size shrinkage used to prevent overfitting.

The models were optimized specifically for the **F1-Score** metric rather than simple accuracy, ensuring a proper balance between Precision and Recall.

---

## 5. Methodology 2: Deep Learning (FT-Transformer)
### 5.1 Theoretical Background
While Neural Networks dominate image and text processing, they have historically struggled against XGBoost on structured, tabular data. The **Feature Tokenizer + Transformer (FT-Transformer)** architecture solves this. 

1. **Feature Tokenizer:** Converts all inputs (both categorical and continuous) into dense vector embeddings (tokens).
2. **Transformer (Self-Attention):** Passes these tokens through multi-head self-attention blocks, allowing the neural network to learn complex, non-linear relationships between different features (e.g., how `Age` uniquely interacts with `Balance` depending on `Geography`).

### 5.2 PyTorch Tabular Implementation
The model was implemented using the `pytorch-tabular` framework. 
* **Architecture Setup:** 3 Attention Blocks, 4 Attention Heads.
* **Training Dynamics:** The model was trained utilizing a GPU. We configured a specific learning rate scheduler and optimized it over 20 epochs. The resulting weights and biases were serialized into a `.ckpt` (checkpoint) directory for production serving.

---

## 6. System Architecture & Engineering
The project transcends a mere Jupyter Notebook by being engineered into a full-stack, decoupled web application.

### 6.1 The Backend (FastAPI)
FastAPI was chosen over Flask/Django for its extreme speed, asynchronous capabilities, and native data validation.
* **Pydantic Validation:** The API defines a `ChurnPredictionRequest` schema. If the frontend sends malformed data (e.g., a string instead of an integer for Age), the API automatically rejects it, preventing server crashes.
* **Dual Inference Engine:** Upon server startup, both the XGBoost `.json` model and the FT-Transformer PyTorch weights are loaded into RAM. When a POST request hits the `/predict` endpoint, the data is pushed through both models sequentially, and a nested JSON response is returned.

### 6.2 The Frontend (React + Vite)
A modern Single Page Application (SPA) was built using React and TypeScript.
* **Glassmorphism UI:** CSS modules were heavily customized to create a premium, translucent design aesthetic.
* **Side-by-Side Rendering:** The UI parses the complex backend JSON payload and dynamically renders two distinct result cards. It uses conditional CSS to display red for high risk and green for low risk, updating dynamically in milliseconds.

### 6.3 Docker Containerization
To eliminate the "it works on my machine" problem, the backend is fully containerized using Docker.
* **The PyTorch Memory Optimization:** A major engineering challenge in cloud deployment is memory limits. Standard PyTorch installations include 3GB+ of NVIDIA CUDA GPU drivers. In the `Dockerfile`, we explicitly route pip to download the lightweight CPU-only wheels (`https://download.pytorch.org/whl/cpu`). This reduces the image size by 80%, allowing it to run flawlessly on free-tier cloud servers.

### 6.4 Cloud Deployment (Render)
* **API Service:** The Dockerized backend is hosted as a Render Web Service.
* **Static Site:** The React app is compiled and hosted globally on Render's CDN. It communicates with the backend via a securely injected `VITE_API_URL` environment variable.
* **Zero Latency Cron Job:** Free cloud tiers "sleep" after 15 minutes of inactivity. We engineered a `/health` endpoint and connected it to an external chronometer (`cron-job.org`) that pings the server every 14 minutes, completely eliminating cold-start latency.

---

## 7. Results & Evaluation Metrics
Because churn is a rare event, relying on Accuracy is highly deceptive. We evaluate the models on their ability to minimize False Positives and False Negatives.

### 7.1 XGBoost Performance
* **Accuracy:** 74.0%
* **ROC-AUC Score:** 85.3%
* **Recall:** 72.2%
* **Precision:** 48.9%
* **F1-Score:** 58.3%

*Analysis:* The `scale_pos_weight` tuning forced XGBoost to be extremely aggressive in catching potential churners (achieving a high Recall of 72.2%). However, this came at the cost of Precision (48.9%), meaning it generated a large number of False Positives (predicting a loyal customer would leave).

### 7.2 FT-Transformer Performance
* **Accuracy:** 86.1%
* **ROC-AUC Score:** 90.2%
* **Recall:** 75.5%
* **Precision:** 62.7%
* **F1-Score:** 68.4%

*Analysis:* The Deep Learning approach demonstrated clear superiority. Through its multi-head attention mechanisms, it learned deeper feature interactions. It maintained a higher Recall (75.5%) while drastically improving Precision (62.7%). 

### 7.3 Feature Importances
According to the XGBoost weight logs, the most critical factors driving customer churn are:
1. **NumOfProducts:** Customers heavily reliant on a specific number of products show massive churn volatility.
2. **Age:** Older demographics showed significantly different retention patterns than younger demographics.
3. **IsActiveMember:** Engagement frequency is a strong predictor of loyalty.

---

## 8. Business Insights & Recommendations
The implementation of the FT-Transformer model provides immediate ROI opportunities for the institution:
1. **Targeted Retention Campaigns:** Instead of blindly offering promotions to the entire user base (which erodes profit margins), the bank can strictly target the subset of customers flagged by the FT-Transformer as >80% risk.
2. **Product Strategy:** Because `NumOfProducts` is the heaviest indicator of churn, the bank should investigate whether specific products (e.g., a specific credit card or loan) are causing customer dissatisfaction.
3. **Age-Specific Interventions:** The strong correlation with `Age` suggests that the bank's digital offerings may be alienating specific age demographics, requiring a UX review of their mobile banking apps.

---

## 9. Conclusion
This project successfully bridges the gap between theoretical data science and production-ready software engineering. By contrasting XGBoost against the FT-Transformer, we empirically validated that while traditional ensemble methods are incredibly fast and reliable, modern tabular Deep Learning architectures can uncover deeper, non-linear relationships that yield superior business metrics. Furthermore, by packaging these complex mathematical models behind a highly accessible React UI and a scalable FastAPI backend, we have created a tool that non-technical stakeholders can use immediately to drive retention strategies.

---

## 10. Tech Stack & Version Requirements
### Core Environment
*   **Operating System:** Cross-Platform (Windows/Linux/macOS) containerized via Docker.
*   **Language:** Python 3.11, TypeScript.

### Machine Learning & Data Processing
*   **XGBoost (v2.0.3):** Core traditional ML algorithm.
*   **PyTorch (v2.14.0+cpu):** Deep learning framework (CPU optimized for cloud deployment).
*   **PyTorch Tabular (v1.2.0):** High-level API for the FT-Transformer.
*   **Pandas & Scikit-learn (v1.4.1):** Data manipulation and metric evaluation.
*   **Weights & Biases (wandb):** Cloud-based experiment tracking and logging.

### Backend Infrastructure
*   **FastAPI (v0.110.0):** Asynchronous API framework.
*   **Uvicorn (v0.27.1):** ASGI web server implementation.
*   **Pydantic:** Data validation schema engine.

### Frontend Infrastructure
*   **React 18:** Component-based UI library.
*   **Vite (v8.3.0):** Next-generation frontend tooling and bundler.
*   **CSS Modules:** Scoped, glassmorphism UI styling.

### Deployment & DevOps
*   **Docker:** Image containerization for the backend.
*   **Render:** Fully managed cloud platform (PaaS) for both Static Sites and Web Services.
*   **cron-job.org:** Keep-alive ping scheduler.
*   **Git & GitHub:** Version control and CI/CD triggers.
