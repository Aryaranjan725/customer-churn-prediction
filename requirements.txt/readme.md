
# Customer Churn Prediction System

## About Project
This is a Machine Learning project that predicts whether a customer is likely to leave a company or stay.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Random Forest
- Streamlit
- Joblib

## Machine Learning Model
Random Forest Classifier is used for customer churn prediction.

## Model Performance
- Accuracy: 75.73%
- Churn Recall: 81% (at threshold 0.45)
- Churn Precision: 51%
- Churn F1-score: 63%

## Features
- Customer information input
- Churn probability prediction
- Stay probability prediction
- Custom classification threshold (0.45)
- Interactive Streamlit interface

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Files
- app.py
- customer_churn_model.pkl
- requirements.txt
- README.md

## Author
Ranjan Arya
