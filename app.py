
import streamlit as st
import pandas as pd
import joblib

# 1. Page settings (must be first Streamlit command)
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# 2. Load trained model
model = joblib.load("customer_churn_model.pkl")

# Selected threshold
THRESHOLD = 0.45

# 3. App heading
st.title("📊 Customer Churn Prediction System")

st.write(
    "Predict whether a customer is likely to leave "
    "the company or stay using Machine Learning."
)

st.info("Prediction Threshold: 0.45")

st.divider()

# 4. Customer details
st.subheader("Enter Customer Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", 0, 100, 12)
    phone = st.selectbox("Phone Service", ["Yes", "No"])

    multiple = st.selectbox(
        "Multiple Lines",
        ["Yes", "No", "No phone service"]
    )

    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )

    security = st.selectbox(
        "Online Security",
        ["Yes", "No", "No internet service"]
    )

    backup = st.selectbox(
        "Online Backup",
        ["Yes", "No", "No internet service"]
    )

with col2:
    device = st.selectbox(
        "Device Protection",
        ["Yes", "No", "No internet service"]
    )

    tech = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )

    tv = st.selectbox(
        "Streaming TV",
        ["Yes", "No", "No internet service"]
    )

    movies = st.selectbox(
        "Streaming Movies",
        ["Yes", "No", "No internet service"]
    )

    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )

    paperless = st.selectbox(
        "Paperless Billing",
        ["Yes", "No"]
    )

    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

    monthly = st.number_input(
        "Monthly Charges",
        min_value=0.0,
        value=70.0
    )

    total = st.number_input(
        "Total Charges",
        min_value=0.0,
        value=840.0
    )

st.divider()

# 5. Prediction
if st.button("🔍 Predict Churn", use_container_width=True):

    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    # Convert categorical values
    customer = pd.get_dummies(
        customer,
        drop_first=True,
        dtype=int
    )

    # Match training columns
    customer = customer.reindex(
        columns=model.feature_names_in_,
        fill_value=0
    )

    # Get churn probability
    churn_probability = model.predict_proba(customer)[0][1]

    stay_probability = 1 - churn_probability

    # Apply selected threshold
    prediction = int(churn_probability >= THRESHOLD)

    # 6. Display results
    st.subheader("Prediction Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.metric(
            "Churn Probability",
            f"{churn_probability * 100:.2f}%"
        )

    with result_col2:
        st.metric(
            "Stay Probability",
            f"{stay_probability * 100:.2f}%"
        )

    if prediction == 1:
        st.error("⚠️ Customer is likely to leave the company!")
    else:
        st.success("✅ Customer is likely to stay with the company!")

    st.caption(
        f"Decision threshold: {THRESHOLD}"
    )

st.divider()

st.caption(
    "Customer Churn Prediction System | "
    "Random Forest Machine Learning Project"
)
