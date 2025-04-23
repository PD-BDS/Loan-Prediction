# Kiva Loan Explorer & Prediction App

This interactive web application allows users to explore Kiva's microloan dataset through a rich visual dashboard and predict loan amounts using a machine learning model. Built with Streamlit, it offers both **Exploratory Data Analysis (EDA)** and **Loan Prediction** capabilities.


## 🚀 Features

### 🔎 Exploratory Data Analysis (EDA)
- Interactive dashboard to filter by:
  - Country
  - Borrower Gender
  - Loan Amount
  - Year
- Visualizations include:
  - Loan sector distribution
  - Term length histogram
  - Monthly loan amounts over time
  - Top 10 countries by average loan amount
  - Gender distribution doughnut chart
- Summary statistics for filtered data

### 🤖 Loan Amount Prediction
- Predict loan amount using:
  - Sector
  - Gender
  - Country
  - Term length
  - Number of lenders
- Uses a pre-trained **XGBoost Regressor**
- Integrated SHAP explanations for model transparency

## 🧪 Tech Stack

- **Python**
- **Pandas** for data manipulation
- **Altair** for interactive visualizations
- **Streamlit** for building the web UI
- **XGBoost** for regression modeling
- **SHAP** for model interpretability
- **Joblib** for model persistence

## 📁 Project Structure

```plaintext
EDA.py                           # EDA Script
Prediction Model.py              # Prediciton interfaces Script
model_xgb.joblib                 # Trained XGBoost model
scaler.joblib                    # Pre-fitted scaler for numeric inputs
ohe.joblib                       # Pre-fitted OneHotEncoder for categorical inputs
kiva_loans.csv                   # Dataset used for analysis and training
train_test_model.ipynb           # Working notebook for training, testing and saving model, scaler, and ohe.
```

## 🛠️ How to Run Locally

1. **Clone this repository**

```bash
git clone https://github.com/yourusername/Loan-Prediction.git
cd Loan-Prediction
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

Make sure your `requirements.txt` includes:
```txt
streamlit
pandas
altair
xgboost
shap
scikit-learn
joblib
```

3. **Run the Streamlit app**

```bash
streamlit run streamlit_app.py
```

## 🧠 Model Explanation with SHAP

The app provides SHAP force plots to help users understand how input features influence the predicted loan amount. These plots are intuitive and highlight the contribution (positive or negative) of each feature to the final prediction.

## 📊 Dataset

The application uses Kiva’s loan dataset, which includes details about loans issued across the globe, including:
- Sector
- Loan amount
- Borrower gender
- Country
- Loan term and more

## 📌 Future Improvements

- Allow CSV uploads for bulk prediction
- Add more models for comparison
- Deploy as a Docker container or Streamlit Cloud app
- Add more in-depth EDA insights (e.g., correlation heatmaps)

## 📜 License

This project is licensed for academic and demonstration purposes. Reach out if you plan to extend it commercially or in research.
