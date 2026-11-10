import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import auth

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="LoanGuard | Credit Risk",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)
# --------------------------------------------------
# AUTHENTICATION
# --------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# --------------------------------------------------
# LOGIN / REGISTRATION
# --------------------------------------------------

if not st.session_state.logged_in:

    st.title("🔐 LoanGuard Login")

    login_tab, register_tab = st.tabs(["Login", "Register"])

    # -----------------------------
    # LOGIN
    # -----------------------------
    with login_tab:

        username = st.text_input("Username", key="login_username")
        password = st.text_input(
            "Password",
            type="password",
            key="login_password"
        )

        if st.button("Login", key="login_button"):

            user = auth.login_user(username, password)

            if user:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # -----------------------------
    # REGISTRATION
    # -----------------------------
    with register_tab:

        new_username = st.text_input(
            "Username",
            key="register_username"
        )

        new_email = st.text_input(
            "Email",
            key="register_email"
        )

        new_password = st.text_input(
            "Password",
            type="password",
            key="register_password"
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            key="confirm_password"
        )

        if st.button("Create Account", key="register_button"):

            if not new_username or not new_email or not new_password:
                st.warning("Please fill in all required fields.")

            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif len(new_password) < 8:
                st.error("Password must be at least 8 characters long.")
            elif not any(char.isupper() for char in new_password):
                st.error("Password must contain at least one uppercase letter.")

            elif not any(char.islower() for char in new_password):
                st.error("Password must contain at least one lowercase letter.")

            elif not any(char.isdigit() for char in new_password):
                st.error("Password must contain at least one number.")


            else:
                success, message = auth.register_user(
                    new_username,
                    new_email,
                    new_password
                )

                if success:
                    st.success(message)
                    st.info("You can now login using your username and password.")
                else:
                    st.error(message)

    st.stop()

# --------------------------------------------------
# LOGOUT
# --------------------------------------------------

if st.session_state.logged_in:
    st.sidebar.write(
        f"Logged in as: {st.session_state.username}"
    )

    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.pop("username", None)
        st.rerun()

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------
st.markdown("""
<style>

.main {
    background-color: #f7f9fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

.hero {
    padding: 1.5rem 2rem;
    border-radius: 15px;
    background: linear-gradient(135deg, #eef4ff, #f8fbff);
    border: 1px solid #dbe7f5;
    margin-bottom: 1.5rem;
}

.hero h1 {
    margin-bottom: 0.3rem;
}

.hero p {
    color: #5f6b7a;
    font-size: 1.05rem;
}

.card {
    padding: 1.3rem;
    border-radius: 14px;
    background: white;
    border: 1px solid #e3e8ef;
    margin-bottom: 1rem;
}

.metric-card {
    padding: 1.5rem;
    border-radius: 14px;
    background: white;
    border: 1px solid #e3e8ef;
    text-align: center;
}

.metric-title {
    color: #667085;
    font-size: 0.9rem;
    margin-bottom: 0.4rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
}

.section-title {
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 1rem;
    margin-bottom: 0.8rem;
}

.footer {
    text-align: center;
    color: #7a8491;
    font-size: 0.85rem;
    margin-top: 2rem;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# HERO SECTION
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>💰 LoanGuard</h1>
    <p>
        Machine Learning based Loan Default Prediction & Credit Risk Analysis
    </p>
</div>
""", unsafe_allow_html=True)

st.write(
    "Assess borrower risk using financial and credit information "
    "with a Random Forest machine learning model."
)

# ---------------------------------------------------------
# SIDEBAR — BORROWER INFORMATION
# ---------------------------------------------------------
st.sidebar.header("👤 Borrower Information")
st.sidebar.caption("Enter borrower details below.")

age = st.sidebar.number_input(
    "Borrower's Age",
    min_value=18,
    max_value=100,
    value=30,
    step=1
)

monthly_income = st.sidebar.number_input(
    "Monthly Income",
    min_value=0.0,
    value=20000.0,
    step=1000.0
)

dependents = st.sidebar.number_input(
    "Number of Dependents",
    min_value=0,
    max_value=20,
    value=3,
    step=1
)

credit_lines = st.sidebar.number_input(
    "Open Credit Lines & Loans",
    min_value=0,
    max_value=100,
    value=3,
    step=1
)

revolving_utilization = st.sidebar.number_input(
    "Revolving Credit Utilization",
    min_value=0.0,
    value=0.80,
    step=0.01,
    format="%.2f"
)

late_30_59 = st.sidebar.number_input(
    "Past-Due Payments: 30–59 Days",
    min_value=0,
    max_value=100,
    value=2,
    step=1
)

debt_ratio = st.sidebar.number_input(
    "Debt Ratio",
    min_value=0.0,
    value=0.70,
    step=0.01,
    format="%.2f"
)

late_90 = st.sidebar.number_input(
    "Past-Due Payments: 90+ Days",
    min_value=0,
    max_value=100,
    value=1,
    step=1
)

real_estate_loans = st.sidebar.number_input(
    "Number of Real Estate Loans",
    min_value=0,
    max_value=100,
    value=1,
    step=1
)

late_60_89 = st.sidebar.number_input(
    "Past-Due Payments: 60–89 Days",
    min_value=0,
    max_value=100,
    value=1,
    step=1
)

predict_button = st.sidebar.button(
    "🔍 Predict Loan Risk",
    use_container_width=True
)

# ---------------------------------------------------------
# MAIN DASHBOARD
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">📊 Credit Risk Assessment</div>',
    unsafe_allow_html=True
)

if predict_button:

    input_data = pd.DataFrame({
        "RevolvingUtilizationOfUnsecuredLines": [revolving_utilization],
        "age": [age],
        "NumberOfTime30-59DaysPastDueNotWorse": [late_30_59],
        "DebtRatio": [debt_ratio],
        "MonthlyIncome": [monthly_income],
        "NumberOfOpenCreditLinesAndLoans": [credit_lines],
        "NumberOfTimes90DaysLate": [late_90],
        "NumberRealEstateLoansOrLines": [real_estate_loans],
        "NumberOfTime60-89DaysPastDueNotWorse": [late_60_89],
        "NumberOfDependents": [dependents]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # -----------------------------------------------------
    # RISK CLASSIFICATION
    # -----------------------------------------------------
    if probability < 0.10:
        risk_level = "🟢 Low"
        risk_message = (
            "Low Risk: The model predicts a low likelihood "
            "of serious delinquency."
        )
    elif probability < 0.30:
        risk_level = "🟠 Medium"
        risk_message = (
            "Medium Risk: The model predicts a moderate likelihood "
            "of serious delinquency."
        )
    else:
        risk_level = "🔴 High"
        risk_message = (
            "High Risk: The model predicts a high likelihood "
            "of serious delinquency."
        )

    # -----------------------------------------------------
    # RESULT CARDS
    # -----------------------------------------------------
    result_col1, result_col2 = st.columns(2)

    with result_col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Estimated Default Probability</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(f"{probability:.2%}"), unsafe_allow_html=True)

    with result_col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Risk Level</div>
            <div class="metric-value">{}</div>
        </div>
        """.format(risk_level), unsafe_allow_html=True)

    st.write("")

    if probability < 0.10:
        st.success(risk_message)
    elif probability < 0.30:
        st.warning(risk_message)
    else:
        st.error(risk_message)

    # -----------------------------------------------------
    # PROBABILITY BAR
    # -----------------------------------------------------
    st.subheader("Risk Probability")

    st.progress(float(probability))

    st.caption(
        f"The model estimates a {probability:.2%} probability "
        "of serious loan delinquency."
    )

    # -----------------------------------------------------
    # RISK FACTORS
    # -----------------------------------------------------
    st.subheader("⚠️ Risk Factors")

    risk_factors = []

    if late_90 > 0:
        risk_factors.append("90+ day late payments detected.")

    if late_60_89 > 0:
        risk_factors.append("60–89 day late payments detected.")

    if late_30_59 > 0:
        risk_factors.append("30–59 day late payments detected.")

    if revolving_utilization > 0.50:
        risk_factors.append("High revolving credit utilization.")

    if debt_ratio > 0.50:
        risk_factors.append("High debt ratio.")

    if risk_factors:
        for factor in risk_factors:
            st.warning(factor)
    else:
        st.success(
            "No major risk factors detected from the provided information."
        )

else:

    st.info(
        "Enter borrower information in the sidebar and click "
        "**Predict Loan Risk** to generate a credit-risk assessment."
    )

# ---------------------------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">🤖 Model Performance</div>',
    unsafe_allow_html=True
)

perf1, perf2, perf3, perf4 = st.columns(4)

with perf1:
    st.metric("Accuracy", f"{metrics['accuracy']:.2%}")

with perf2:
    st.metric("Precision", f"{metrics['precision']:.2%}")

with perf3:
    st.metric("Class 1 Recall", f"{metrics['recall']:.2%}")

with perf4:
    st.metric("F1-Score", f"{metrics['f1_score']:.2%}")

st.caption(
    "Performance metrics are based on the test dataset used "
    "during model evaluation."
)

# ---------------------------------------------------------
# DISCLAIMER
# ---------------------------------------------------------
st.info(
    "Disclaimer: This application is intended for educational and "
    "demonstration purposes only. The prediction should not be used "
    "as the sole basis for real-world lending decisions."
)

st.markdown("""
<div class="footer">
    LoanGuard • Machine Learning Credit Risk Analysis
</div>
""", unsafe_allow_html=True)