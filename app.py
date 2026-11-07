
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="💰",
    layout="wide"
)


# --------------------------------------------------
# Load Model and Metrics
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "loan_default_model.pkl"
METRICS_PATH = BASE_DIR / "models" / "model_metrics.pkl"


try:
    model = joblib.load(MODEL_PATH)
    metrics = joblib.load(METRICS_PATH)
except FileNotFoundError:
    st.error(
        "Model files not found. Please make sure the required files "
        "are present inside the 'models' folder."
    )
    st.stop()


# --------------------------------------------------
# Application Header
# --------------------------------------------------

st.title("💰 Loan Default Prediction & Credit Risk Analysis")

st.write(
    "Enter the borrower's financial and credit information "
    "to estimate the likelihood of serious loan delinquency."
)


# --------------------------------------------------
# Borrower Information
# --------------------------------------------------

st.header("Borrower Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "Borrower's Age",
        min_value=18,
        max_value=100,
        value=30,
        step=1,
        key="age"
    )

    monthly_income = st.number_input(
        "Monthly Income",
        min_value=0.0,
        value=20000.0,
        step=1000.0,
        key="monthly_income"
    )

    dependents = st.number_input(
        "Number of Dependents",
        min_value=0,
        max_value=20,
        value=3,
        step=1,
        key="dependents"
    )

    credit_lines = st.number_input(
        "Number of Open Credit Lines and Loans",
        min_value=0,
        max_value=100,
        value=3,
        step=1,
        key="credit_lines"
    )

    revolving_utilization = st.number_input(
        "Revolving Credit Utilization",
        min_value=0.0,
        value=0.80,
        step=0.01,
        format="%.2f",
        key="revolving_utilization"
    )


with col2:

    late_30_59 = st.number_input(
        "Past-Due Payments: 30–59 Days",
        min_value=0,
        max_value=100,
        value=2,
        step=1,
        key="late_30_59"
    )

    debt_ratio = st.number_input(
        "Debt Ratio",
        min_value=0.0,
        value=0.70,
        step=0.01,
        format="%.2f",
        key="debt_ratio"
    )

    late_90 = st.number_input(
        "Past-Due Payments: 90+ Days",
        min_value=0,
        max_value=100,
        value=1,
        step=1,
        key="late_90"
    )

    real_estate_loans = st.number_input(
        "Number of Real Estate Loans",
        min_value=0,
        max_value=100,
        value=1,
        step=1,
        key="real_estate_loans"
    )

    late_60_89 = st.number_input(
        "Past-Due Payments: 60–89 Days",
        min_value=0,
        max_value=100,
        value=1,
        step=1,
        key="late_60_89"
    )


# --------------------------------------------------
# Prediction Button
# --------------------------------------------------

if st.button("Predict Loan Default Risk"):

    # Create input DataFrame
    # The column names and order must match
    # the features used during model training.

    input_data = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [
            revolving_utilization
        ],
        "age": [
            age
        ],
        "NumberOfTime30-59DaysPastDueNotWorse": [
            late_30_59
        ],
        "DebtRatio": [
            debt_ratio
        ],
        "MonthlyIncome": [
            monthly_income
        ],
        "NumberOfOpenCreditLinesAndLoans": [
            credit_lines
        ],
        "NumberOfTimes90DaysLate": [
            late_90
        ],
        "NumberRealEstateLoansOrLines": [
            real_estate_loans
        ],
        "NumberOfTime60-89DaysPastDueNotWorse": [
            late_60_89
        ],
        "NumberOfDependents": [
            dependents
        ]
    })


    # --------------------------------------------------
    # Make Prediction
    # --------------------------------------------------

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]


    # --------------------------------------------------
    # Risk Level
    # --------------------------------------------------

    if probability < 0.10:
        risk_level = "🟢 Low"

    elif probability < 0.30:
        risk_level = "🟠 Medium"

    else:
        risk_level = "🔴 High"


    # --------------------------------------------------
    # Display Prediction Result
    # --------------------------------------------------

    st.header("Loan Default Risk Result")

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            label="Estimated Default Risk",
            value=f"{probability:.2%}"
        )

    with result_col2:

        st.metric(
            label="Risk Level",
            value=risk_level
        )


    # --------------------------------------------------
    # Risk Message
    # --------------------------------------------------

    # The message is based on the same probability
    # thresholds used for the Low/Medium/High risk level.

    if probability < 0.10:

        st.success(
            "Low Risk: The model predicts a low likelihood "
            "of serious delinquency."
        )

    elif probability < 0.30:

        st.warning(
            "Medium Risk: The model predicts a moderate likelihood "
            "of serious delinquency."
        )

    else:

        st.error(
            "High Risk: The model predicts a high likelihood "
            "of serious delinquency."
        )


    # --------------------------------------------------
    # Risk Factors
    # --------------------------------------------------

    st.subheader("⚠️ Risk Factors")

    risk_factors = []

    if late_90 > 0:
        risk_factors.append(
            "90+ day late payments detected."
        )

    if late_60_89 > 0:
        risk_factors.append(
            "60–89 day late payments detected."
        )

    if late_30_59 > 0:
        risk_factors.append(
            "30–59 day late payments detected."
        )

    if revolving_utilization > 0.50:
        risk_factors.append(
            "High revolving credit utilization."
        )

    if debt_ratio > 0.50:
        risk_factors.append(
            "High debt ratio."
        )


    if risk_factors:

        for factor in risk_factors:
            st.warning(factor)

    else:

        st.success(
            "No major risk factors detected from the provided information."
        )


    # --------------------------------------------------
    # Probability Progress Bar
    # --------------------------------------------------

    st.progress(float(probability))

    st.caption(
        f"Estimated default probability: {probability:.2%}"
    )


# --------------------------------------------------
# Model Performance
# --------------------------------------------------

st.subheader("📊 Model Performance")

st.write("Random Forest Model")

st.write(
    f"Accuracy: {metrics['accuracy']:.2%}"
)

st.write(
    f"Precision: {metrics['precision']:.2%}"
)

st.write(
    f"Class 1 Recall: {metrics['recall']:.2%}"
)

st.write(
    f"F1-Score: {metrics['f1_score']:.2%}"
)

st.caption(
    "Performance metrics are based on the test dataset used "
    "during model evaluation."
)


# --------------------------------------------------
# Disclaimer
# --------------------------------------------------

st.info(
    "Disclaimer: This application is intended for educational "
    "and demonstration purposes only. The prediction should "
    "not be used as the sole basis for real-world lending decisions."
)
