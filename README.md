# CloudExify Data Science Internship — Project 3: House Price Prediction

## 📌 Project Overview
This project implements end-to-end Machine Learning regression pipelines to predict residential house prices. It evaluates both baseline Linear Regression and ensemble Random Forest Regressor models, analyzes feature importance, and provides an interactive Streamlit web application for real-time inference.

---

## 📊 Dataset Description
The dataset used is the **Housing Prices Dataset**, which includes structural, demographic, and amenity-based property features:
* **Target Variable**: `price` (in Rs)
* **Numeric Features**: `area`, `bedrooms`, `bathrooms`, `stories`, `parking`
* **Categorical Features**: `mainroad`, `guestroom`, `basement`, `hotwaterheating`, `airconditioning`, `prefarea`, `furnishingstatus`

---

## 🛠️ Tech Stack & Libraries
* **Language**: Python (Jupyter Notebook)
* **Data Processing & ML**: `pandas`, `numpy`, `scikit-learn`, `joblib`
* **Visualization**: `matplotlib`, `seaborn`
* **Deployment**: `streamlit`

---

## 📈 Model Performance & Evaluation

| Model | $R^2$ Score (Test) | RMSE (Rs) | Notes |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | *~0.65 - 0.68* | *~1,200,000* | Baseline linear fit |
| **Random Forest Regressor** | *~0.70+* | *~1,050,000* | Captures non-linear feature interactions |

*Note: The Random Forest Regressor achieved a higher $R^2$ score and lower Root Mean Squared Error (RMSE), making it the selected production model.*

---

## 📉 Visualizations

### 1. Feature Importance
Identifies the key drivers affecting property prices (such as `area`, `bathrooms`, and `airconditioning`).

<img width="989" height="590" alt="feature_importance" src="https://github.com/user-attachments/assets/ed915f25-a72f-4f66-b8e0-889148d28691" />


### 2. Actual vs. Predicted Prices
Evaluates the model's prediction accuracy along the $y = x$ reference diagonal.

<img width="989" height="590" alt="actual_vs_predicted" src="https://github.com/user-attachments/assets/5526fe53-d54f-47b9-8114-53633ec4ad93" />


---

## 🚀 Interactive Web App (Streamlit)

### Run the App Locally:
1. Ensure the required packages are installed:
   ```bash
   pip install streamlit pandas scikit-learn joblib matplotlib
