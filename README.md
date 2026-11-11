# Customer Churn Prediction System

## About Project
This is a Machine Learning project that predicts whether a customer is likely to leave a company or stay.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Streamlit
- Joblib

## Machine Learning Model
Random Forest Classifier is used to predict customer churn.

## Model Performance
- Accuracy: 75.73%
- Churn Recall: 81%
- Churn Precision: 51%
- Churn F1-score: 63%

*Note: These are the current test results at a threshold of 0.45.*

## Features
- Customer information input
- Churn probability prediction
- Stay probability prediction
- Custom prediction threshold (0.45)
- Interactive Streamlit interface

## Project Screenshots

### Screenshot 1
![Customer Churn App](Screenshot%202026-11-11%20222616.png)

### Screenshot 2
![Customer Churn Prediction](Screenshot%202026-11-11%20222824.png)

### Screenshot 3
![Prediction Result](Screenshot%202026-11-11%20222909.png)

## How to Run

Install the required libraries:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Live Demo

[Customer Churn Prediction App](https://customer-churn-prediction-j8xybmddmikqphxr9tfatp.streamlit.app/)

## Author
Ranjan Arya
