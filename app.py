import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# --------------------------------------------------
# Load the trained machine learning model
# --------------------------------------------------

model_path = Path(__file__).parent / "models" / "loan_default_model.pkl"
model = joblib.load(model_path)


# --------------------------------------------------
# Configure Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="🏦",
    layout="centered"
)


# --------------------------------------------------
# Application title and description
# --------------------------------------------------

st.title("🏦 Loan Default Prediction System")

st.write(
    "Predict the likelihood of serious loan delinquency "
    "using a Machine Learning model."
)

st.divider()


# --------------------------------------------------
# Borrower Information
# --------------------------------------------------

st.subheader("Borrower Information")


age = st.number_input(
    "Borrower's Age",
    min_value=18,
    max_value=100,
    value=30
)

monthly_income = st.number_input(
    "Monthly Income",
    min_value=0,
    value=30000
)

dependents = st.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=0
)

credit_lines = st.number_input(
    "Number of Open Credit Lines and Loans",
    min_value=0,
    value=5
)

revolving_utilization = st.number_input(
    "Revolving Credit Utilization",
    min_value=0.0,
    value=0.30
)

late_30_59 = st.number_input(
    "Past-Due Payments: 30–59 Days",
    min_value=0,
    value=0
)

debt_ratio = st.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.30
)

late_90 = st.number_input(
    "Past-Due Payments: 90+ Days",
    min_value=0,
    value=0
)

real_estate_loans = st.number_input(
    "Number of Real Estate Loans",
    min_value=0,
    value=1
)

late_60_89 = st.number_input(
    "Past-Due Payments: 60–89 Days",
    min_value=0,
    value=0
)


st.divider()


# --------------------------------------------------
# Predict Loan Default Risk
# --------------------------------------------------

if st.button("Predict Loan Default Risk"):

    # Prepare borrower data
    input_data = pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": revolving_utilization,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late_30_59,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": monthly_income,
        "NumberOfOpenCreditLinesAndLoans": credit_lines,
        "NumberOfTimes90DaysLate": late_90,
        "NumberRealEstateLoansOrLines": real_estate_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late_60_89,
        "NumberOfDependents": dependents
    }])

    # Generate prediction
    prediction = model.predict(input_data)[0]

    # Calculate probability of serious delinquency
    probability = model.predict_proba(input_data)[0][1]


    # --------------------------------------------------
    # Display Prediction Result
    # --------------------------------------------------

    st.subheader("Loan Default Risk Result")


    if prediction == 1:

        st.error(
            "High Risk: The model predicts possible serious delinquency."
        )

    else:

        st.success(
            "Lower Risk: The model predicts a lower likelihood "
            "of serious delinquency."
        )


    # Display probability
    st.write(
        f"Estimated default risk: {probability:.2%}"
    )


    # --------------------------------------------------
    # NEW FEATURE: Risk Level
    # --------------------------------------------------

    if probability < 0.10:

        st.success("🟢 Risk Level: Low")

    elif probability < 0.30:

        st.warning("🟠 Risk Level: Medium")

    else:

        st.error("🔴 Risk Level: High")


    # Probability progress bar
    st.progress(float(probability))


    # Disclaimer
    st.caption(
        "This percentage is a model estimate, not a guarantee "
        "of whether a borrower will default."
    )