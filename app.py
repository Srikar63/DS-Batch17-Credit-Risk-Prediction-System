"""
Credit Risk Prediction System - Main Streamlit Dashboard.
Production-grade Explainable AI (XAI), Historical Data Analysis,
and Responsible Decision Support for Credit Risk Underwriting.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json

from src.config import (
    CSV_DATA_PATH,
    TARGET_COL,
    TARGET_LABELS,
    NUMERICAL_COLS,
    CATEGORICAL_COLS,
    ALL_FEATURE_COLS,
    NUMERICAL_SPECS,
    CATEGORICAL_OPTIONS,
    RISK_THRESHOLD_LOW,
    RISK_THRESHOLD_HIGH,
    classify_risk_tier
)
from src.data_processor import (
    load_data,
    get_dataset_profile,
    transform_single_applicant
)
from src.model_trainer import (
    train_and_evaluate_all_models,
    load_trained_artifacts
)
from src.xai_engine import XAIEngine
from src.historical_matcher import HistoricalMatcher
from src.fairness_evaluator import FairnessEvaluator
from src.history_store import HistoryStore

# Page Configuration
st.set_page_config(
    page_title="Credit Risk Intelligence | XAI Decision Support",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        font-size: 1.05rem;
        color: #64748b;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }

    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .kpi-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin-top: 0.25rem;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.1rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
        letter-spacing: 0.025em;
    }

    .banner-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
        font-size: 0.9rem;
        color: #334155;
    }

    .tag-increasing {
        background-color: #ffe4e6;
        color: #be123c;
        border: 1px solid #fecdd3;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }

    .tag-decreasing {
        background-color: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
        padding: 0.25rem 0.6rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.25rem;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading & artifact retrieval
@st.cache_data
def get_dataset():
    return load_data()

@st.cache_resource
def get_artifacts():
    return load_trained_artifacts()

# Load core resources
df_data = get_dataset()
artifacts = get_artifacts()

all_models = artifacts["all_models"]
preprocessor = artifacts["preprocessor"]
metrics_summary = artifacts["metrics_summary"]
confusion_matrices = artifacts["confusion_matrices"]
roc_curves_data = artifacts["roc_curves_data"]
train_data_bundle = artifacts["train_data"]

X_train_raw = train_data_bundle["X_train"]
y_train_raw = train_data_bundle["y_train"]
X_train_proc = train_data_bundle["X_train_proc"]
X_test_proc = train_data_bundle["X_test_proc"]
y_test = train_data_bundle["y_test"]
feature_names = train_data_bundle["feature_names"]

# Initialize helper engines
history_store = HistoryStore()
historical_matcher = HistoricalMatcher(
    X_train_raw=X_train_raw,
    y_train_raw=y_train_raw,
    X_train_proc=X_train_proc,
    preprocessor=preprocessor
)

# Sidebar Navigation & Model Selector
st.sidebar.markdown("## 💳 Credit Risk System")
st.sidebar.caption("Explainable AI & Decision Support")

active_model_name = st.sidebar.selectbox(
    "Active Scoring Model",
    options=["Random Forest", "Logistic Regression", "Support Vector Machine", "Decision Tree"],
    index=0,
    help="Select the underlying machine learning model used to evaluate applicants and generate XAI attributions."
)

active_model = all_models[active_model_name]

# Initialize XAI Engine with active model
xai_engine = XAIEngine(
    model=active_model,
    feature_names=feature_names,
    X_train_proc=X_train_proc,
    model_name=active_model_name
)

# Navigation Menu
nav_choice = st.sidebar.radio(
    "Navigation Menu",
    [
        "📊 Executive Dashboard",
        "📝 Applicant Assessment",
        "🔍 Explainable AI (XAI)",
        "🏛️ Historical Analysis",
        "🔄 What-If Analysis",
        "⚖️ Model Comparison",
        "🛡️ Responsible AI & Fairness",
        "📜 Prediction History"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Risk Thresholds")
st.sidebar.markdown(rf"""
- **Low Risk:** $< {RISK_THRESHOLD_LOW:.0%}$
- **Medium Risk:** ${RISK_THRESHOLD_LOW:.0%} - {RISK_THRESHOLD_HIGH:.0%}$
- **High Risk:** $\ge {RISK_THRESHOLD_HIGH:.0%}$
""")
st.sidebar.caption("Thresholds calibrated to balance 30% baseline default rate with 5:1 loss asymmetry.")

# System Advisory Banner
st.markdown("""
<div class="banner-box">
    <strong>⚖️ Responsible Lending Decision Support:</strong> This system provides probabilistic risk estimates and explainability insights for qualified underwriting personnel. It does not replace credit authority review or guarantee loan outcomes.
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. EXECUTIVE DASHBOARD
# -------------------------------------------------------------
if nav_choice == "📊 Executive Dashboard":
    st.markdown('<div class="main-title">Credit Risk Executive Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Portfolio overview, class distributions, demographic profiles, and model integrity.</div>', unsafe_allow_html=True)

    profile = get_dataset_profile(df_data)

    # Top KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Portfolio Records</div>
            <div class="kpi-value">{profile['total_records']:,}</div>
            <div class="kpi-sub">Clean Historical Borrowers</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Historical Default Rate</div>
            <div class="kpi-value">{profile['default_rate']:.1%}</div>
            <div class="kpi-sub">{profile['class_distribution'].get(1, 0)} Defaulted Borrowers</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Good Credit Share</div>
            <div class="kpi-value">{(1 - profile['default_rate']):.1%}</div>
            <div class="kpi-sub">{profile['class_distribution'].get(0, 0)} Repaid Loans</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        best_roc = metrics_summary.get(active_model_name, {}).get("ROC-AUC", 0.81)
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{active_model_name} ROC-AUC</div>
            <div class="kpi-value">{best_roc:.3f}</div>
            <div class="kpi-sub">Evaluated on Out-of-Sample Test Set</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualizations
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🎯 Target Risk Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = ["Repaid / Good (Class 0)", "Default / Bad (Class 1)"]
        counts = [profile['class_distribution'].get(0, 700), profile['class_distribution'].get(1, 300)]
        colors = ["#10b981", "#ef4444"]
        wedges, texts, autotexts = ax.pie(
            counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140,
            wedgeprops=dict(width=0.45, edgecolor='w')
        )
        for t in texts:
            t.set_fontsize(10)
        for at in autotexts:
            at.set_fontsize(10)
            at.set_weight('bold')
        ax.set_title("Historical Portfolio Repayment Outcome", fontsize=11, fontweight='bold')
        st.pyplot(fig)
        plt.close(fig)

    with chart_col2:
        st.markdown("#### 🏦 Default Rate by Checking Account Status")
        checking_df = df_data.groupby("checking_status")["credit_risk"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=checking_df, x="checking_status", y="credit_risk", hue="checking_status", palette="Blues_r", legend=False, ax=ax)
        ax.set_ylabel("Default Rate", fontsize=10)
        ax.set_xlabel("Checking Account Balance Tier", fontsize=10)
        ax.set_title("Empirical Default Rate across Liquidity Tiers", fontsize=11, fontweight='bold')
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{height:.1%}', (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9, xytext=(0, 2), textcoords='offset points')
        ax.set_ylim(0, max(checking_df["credit_risk"]) * 1.25)
        st.pyplot(fig)
        plt.close(fig)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown("#### ⏳ Loan Duration vs Credit Amount by Risk")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_data,
            x="duration",
            y="credit_amount",
            hue="credit_risk",
            palette={0: "#10b981", 1: "#ef4444"},
            alpha=0.6,
            ax=ax
        )
        ax.set_xlabel("Duration (Months)", fontsize=10)
        ax.set_ylabel("Credit Amount (DM)", fontsize=10)
        ax.set_title("Principal Amount vs Duration by Risk Outcome", fontsize=11, fontweight='bold')
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, ["Repaid", "Default"], title="Outcome")
        st.pyplot(fig)
        plt.close(fig)

    with chart_col4:
        st.markdown("#### 📜 Credit History vs Observed Risk")
        history_df = df_data.groupby("credit_history")["credit_risk"].agg(["mean", "count"]).reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=history_df, y="credit_history", x="mean", hue="credit_history", palette="magma", legend=False, ax=ax)
        ax.set_xlabel("Default Probability", fontsize=10)
        ax.set_ylabel("", fontsize=10)
        ax.set_title("Default Rate across Credit History Profiles", fontsize=11, fontweight='bold')
        for p in ax.patches:
            w = p.get_width()
            ax.annotate(f'{w:.1%}', (w, p.get_y() + p.get_height()/2.),
                        ha='left', va='center', fontsize=9, xytext=(3, 0), textcoords='offset points')
        st.pyplot(fig)
        plt.close(fig)

    # Dataset profile summary table
    st.markdown("---")
    st.markdown("#### 📋 Dataset Technical Profile & Schema")
    with st.expander("View Complete Dataset Schema and Profiling Details", expanded=False):
        schema_df = pd.DataFrame({
            "Column Name": list(df_data.columns),
            "Data Type": [str(t) for t in df_data.dtypes.values],
            "Missing Values": [df_data[c].isnull().sum() for c in df_data.columns],
            "Unique Values": [df_data[c].nunique() for c in df_data.columns],
            "Sample Values": [str(df_data[c].dropna().unique()[:3].tolist()) for c in df_data.columns]
        })
        st.dataframe(schema_df, use_container_width=True)

# -------------------------------------------------------------
# 2. APPLICANT ASSESSMENT
# -------------------------------------------------------------
elif nav_choice == "📝 Applicant Assessment":
    st.markdown('<div class="main-title">Applicant Credit Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter borrower profile details to receive probabilistic risk categorization, approval support, and instant XAI factors.</div>', unsafe_allow_html=True)

    # Preset profiles for easy evaluation
    st.markdown("##### ⚡ Quick Load Benchmark Profiles")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    if "applicant_input" not in st.session_state:
        st.session_state.applicant_input = {
            "duration": 24,
            "credit_amount": 3500,
            "installment_rate": 3,
            "residence_since": 3,
            "age": 36,
            "existing_credits": 1,
            "num_dependents": 1,
            "checking_status": ">= 200 DM",
            "credit_history": "existing credits paid duly till now",
            "purpose": "car (new)",
            "savings_status": "500 to < 1000 DM",
            "employment": "4 to < 7 years",
            "personal_status_sex": "male : single",
            "other_parties": "none",
            "property_magnitude": "real estate",
            "other_payment_plans": "none",
            "housing": "own",
            "job": "skilled employee/official",
            "telephone": "yes",
            "foreign_worker": "no"
        }

    with preset_col1:
        if st.button("🟢 Load Prime / Low-Risk Profile", use_container_width=True):
            st.session_state.applicant_input = {
                "duration": 12,
                "credit_amount": 1500,
                "installment_rate": 2,
                "residence_since": 4,
                "age": 42,
                "existing_credits": 2,
                "num_dependents": 1,
                "checking_status": ">= 200 DM",
                "credit_history": "all credits at this bank paid duly",
                "purpose": "radio/television",
                "savings_status": ">= 1000 DM",
                "employment": ">= 7 years",
                "personal_status_sex": "male : single",
                "other_parties": "none",
                "property_magnitude": "real estate",
                "other_payment_plans": "none",
                "housing": "own",
                "job": "management/self-employed/highly qualified",
                "telephone": "yes",
                "foreign_worker": "no"
            }
            st.rerun()

    with preset_col2:
        if st.button("🟡 Load Moderate-Risk Profile", use_container_width=True):
            st.session_state.applicant_input = {
                "duration": 24,
                "credit_amount": 3800,
                "installment_rate": 3,
                "residence_since": 2,
                "age": 29,
                "existing_credits": 1,
                "num_dependents": 1,
                "checking_status": "0 to < 200 DM",
                "credit_history": "existing credits paid duly till now",
                "purpose": "furniture/equipment",
                "savings_status": "100 to < 500 DM",
                "employment": "1 to < 4 years",
                "personal_status_sex": "female : single",
                "other_parties": "none",
                "property_magnitude": "car or other",
                "other_payment_plans": "none",
                "housing": "rent",
                "job": "skilled employee/official",
                "telephone": "none",
                "foreign_worker": "yes"
            }
            st.rerun()

    with preset_col3:
        if st.button("🔴 Load High-Risk Profile", use_container_width=True):
            st.session_state.applicant_input = {
                "duration": 48,
                "credit_amount": 9500,
                "installment_rate": 4,
                "residence_since": 1,
                "age": 23,
                "existing_credits": 1,
                "num_dependents": 1,
                "checking_status": "< 0 DM",
                "credit_history": "critical account / other credits existing",
                "purpose": "business",
                "savings_status": "< 100 DM",
                "employment": "< 1 year",
                "personal_status_sex": "male : divorced/separated",
                "other_parties": "none",
                "property_magnitude": "unknown / no property",
                "other_payment_plans": "bank",
                "housing": "rent",
                "job": "unskilled - resident",
                "telephone": "none",
                "foreign_worker": "yes"
            }
            st.rerun()

    st.markdown("---")

    # Input Form organized into logical tabs
    tab_loan, tab_bank, tab_profile = st.tabs(["💳 Loan Terms", "🏦 Liquidity & Banking", "👤 Applicant Demographics & Collateral"])

    current_data = st.session_state.applicant_input

    with tab_loan:
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            duration = st.slider(
                "Loan Duration (Months)",
                min_value=NUMERICAL_SPECS["duration"]["min"],
                max_value=NUMERICAL_SPECS["duration"]["max"],
                value=int(current_data["duration"]),
                step=1
            )
            credit_amount = st.slider(
                "Requested Credit Amount (DM)",
                min_value=NUMERICAL_SPECS["credit_amount"]["min"],
                max_value=NUMERICAL_SPECS["credit_amount"]["max"],
                value=int(current_data["credit_amount"]),
                step=100
            )
        with col_l2:
            purpose = st.selectbox(
                "Loan Purpose",
                options=CATEGORICAL_OPTIONS["purpose"],
                index=CATEGORICAL_OPTIONS["purpose"].index(current_data["purpose"])
            )
            installment_rate = st.selectbox(
                "Installment Rate (% of Disposable Income)",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(int(current_data["installment_rate"]))
            )

    with tab_bank:
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            checking_status = st.selectbox(
                "Status of Existing Checking Account",
                options=CATEGORICAL_OPTIONS["checking_status"],
                index=CATEGORICAL_OPTIONS["checking_status"].index(current_data["checking_status"])
            )
            savings_status = st.selectbox(
                "Savings Account / Liquid Bonds",
                options=CATEGORICAL_OPTIONS["savings_status"],
                index=CATEGORICAL_OPTIONS["savings_status"].index(current_data["savings_status"])
            )
            credit_history = st.selectbox(
                "Credit History at Banks",
                options=CATEGORICAL_OPTIONS["credit_history"],
                index=CATEGORICAL_OPTIONS["credit_history"].index(current_data["credit_history"])
            )
        with col_b2:
            existing_credits = st.selectbox(
                "Number of Existing Credits at this Bank",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(int(current_data["existing_credits"]))
            )
            other_payment_plans = st.selectbox(
                "Other Installment Plans",
                options=CATEGORICAL_OPTIONS["other_payment_plans"],
                index=CATEGORICAL_OPTIONS["other_payment_plans"].index(current_data["other_payment_plans"])
            )
            other_parties = st.selectbox(
                "Other Debtors / Guarantors",
                options=CATEGORICAL_OPTIONS["other_parties"],
                index=CATEGORICAL_OPTIONS["other_parties"].index(current_data["other_parties"])
            )

    with tab_profile:
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            employment = st.selectbox(
                "Present Employment Duration",
                options=CATEGORICAL_OPTIONS["employment"],
                index=CATEGORICAL_OPTIONS["employment"].index(current_data["employment"])
            )
            age = st.slider(
                "Applicant Age (Years)",
                min_value=NUMERICAL_SPECS["age"]["min"],
                max_value=NUMERICAL_SPECS["age"]["max"],
                value=int(current_data["age"]),
                step=1
            )
            personal_status_sex = st.selectbox(
                "Personal Status & Gender",
                options=CATEGORICAL_OPTIONS["personal_status_sex"],
                index=CATEGORICAL_OPTIONS["personal_status_sex"].index(current_data["personal_status_sex"])
            )
            residence_since = st.selectbox(
                "Years at Current Residence",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(int(current_data["residence_since"]))
            )
        with col_p2:
            property_magnitude = st.selectbox(
                "Property & Asset Ownership",
                options=CATEGORICAL_OPTIONS["property_magnitude"],
                index=CATEGORICAL_OPTIONS["property_magnitude"].index(current_data["property_magnitude"])
            )
            housing = st.selectbox(
                "Housing Arrangement",
                options=CATEGORICAL_OPTIONS["housing"],
                index=CATEGORICAL_OPTIONS["housing"].index(current_data["housing"])
            )
            job = st.selectbox(
                "Employment Type / Job Category",
                options=CATEGORICAL_OPTIONS["job"],
                index=CATEGORICAL_OPTIONS["job"].index(current_data["job"])
            )
            num_dependents = st.selectbox(
                "Number of Dependents Liable for Maintenance",
                options=[1, 2],
                index=[1, 2].index(int(current_data["num_dependents"]))
            )
            col_sub1, col_sub2 = st.columns(2)
            with col_sub1:
                telephone = st.selectbox(
                    "Telephone Registered",
                    options=CATEGORICAL_OPTIONS["telephone"],
                    index=CATEGORICAL_OPTIONS["telephone"].index(current_data["telephone"])
                )
            with col_sub2:
                foreign_worker = st.selectbox(
                    "Foreign Worker Status",
                    options=CATEGORICAL_OPTIONS["foreign_worker"],
                    index=CATEGORICAL_OPTIONS["foreign_worker"].index(current_data["foreign_worker"])
                )

    # Pack applicant dict
    applicant_payload = {
        "duration": duration,
        "credit_amount": credit_amount,
        "installment_rate": installment_rate,
        "residence_since": residence_since,
        "age": age,
        "existing_credits": existing_credits,
        "num_dependents": num_dependents,
        "checking_status": checking_status,
        "credit_history": credit_history,
        "purpose": purpose,
        "savings_status": savings_status,
        "employment": employment,
        "personal_status_sex": personal_status_sex,
        "other_parties": other_parties,
        "property_magnitude": property_magnitude,
        "other_payment_plans": other_payment_plans,
        "housing": housing,
        "job": job,
        "telephone": telephone,
        "foreign_worker": foreign_worker
    }
    st.session_state.applicant_input = applicant_payload

    # Predict button
    st.markdown("<br>", unsafe_allow_html=True)
    assess_btn = st.button("🚀 Evaluate Applicant Credit Risk", type="primary", use_container_width=True)

    # Perform evaluation
    applicant_proc = transform_single_applicant(applicant_payload, preprocessor)
    default_prob = float(active_model.predict_proba(applicant_proc)[:, 1][0]) if hasattr(active_model, "predict_proba") else float(active_model.predict(applicant_proc)[0])
    risk_tier, badge_color, recommendation = classify_risk_tier(default_prob)

    # Compute SHAP factors
    shap_results = xai_engine.explain_instance_shap(applicant_proc, top_n=4)
    top_inc = shap_results["top_increasing"]
    top_dec = shap_results["top_decreasing"]

    # Store in history if button pressed
    if assess_btn:
        top_factor_names = [f[0] for f in top_inc[:3]]
        history_store.log_assessment(applicant_payload, risk_tier, default_prob, top_factor_names)
        st.toast(f"Assessment logged: {risk_tier} ({default_prob:.1%} Default Probability)", icon="✅")

    # Assessment Results Display
    st.markdown("---")
    st.markdown("### 📊 Assessment Decision Output")

    res_col1, res_col2 = st.columns([1, 1.4])

    with res_col1:
        st.markdown(f"""
        <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1.5rem; text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase;">Predicted Risk Category</div>
            <div style="margin: 0.75rem 0;">
                <span class="risk-badge" style="background-color: {badge_color}22; color: {badge_color}; border: 2px solid {badge_color}; font-size: 1.35rem;">
                    {risk_tier.upper()}
                </span>
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #0f172a; margin-top: 0.5rem;">
                {default_prob:.1%}
            </div>
            <div style="font-size: 0.85rem; color: #64748b; font-weight: 500;">
                Estimated Probability of Default
            </div>
            <div style="margin-top: 1rem; text-align: left; background: white; border-radius: 8px; padding: 0.8rem 1rem; border: 1px solid #e2e8f0;">
                <div style="font-size: 0.75rem; font-weight: 700; color: #94a3b8; text-transform: uppercase;">Decision Support Guidance</div>
                <div style="font-size: 0.88rem; color: #334155; margin-top: 0.25rem;">{recommendation}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### 🔍 Primary Influence Drivers (XAI Summary)")
        st.markdown("**Top Factors Increasing Risk (Red Flags):**")
        if top_inc:
            tags_html = "".join([f'<span class="tag-increasing">▲ {xai_engine._humanize_feature_name(f[0])} (+{f[1]:.3f})</span>' for f in top_inc[:3]])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.write("No major risk-increasing deviations detected.")

        st.markdown("<br>**Top Factors Decreasing Risk (Protective Merits):**", unsafe_allow_html=True)
        if top_dec:
            tags_html = "".join([f'<span class="tag-decreasing">▼ {xai_engine._humanize_feature_name(f[0])} ({f[1]:.3f})</span>' for f in top_dec[:3]])
            st.markdown(tags_html, unsafe_allow_html=True)
        else:
            st.write("No major risk-mitigating features detected.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 To inspect the complete waterfall breakdown, visit the **🔍 Explainable AI (XAI)** tab. To test adjustments, visit **🔄 What-If Analysis**.")

# -------------------------------------------------------------
# 3. EXPLAINABLE AI (XAI)
# -------------------------------------------------------------
elif nav_choice == "🔍 Explainable AI (XAI)":
    st.markdown('<div class="main-title">Explainable AI (XAI) Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Local applicant-level SHAP attributions, dynamic natural language narratives, LIME cross-validation, and global feature importance.</div>', unsafe_allow_html=True)

    if "applicant_input" not in st.session_state:
        st.warning("Please configure an applicant in the '📝 Applicant Assessment' tab first.")
        st.stop()

    applicant_payload = st.session_state.applicant_input
    applicant_proc = transform_single_applicant(applicant_payload, preprocessor)
    default_prob = float(active_model.predict_proba(applicant_proc)[:, 1][0]) if hasattr(active_model, "predict_proba") else float(active_model.predict(applicant_proc)[0])
    risk_tier, badge_color, _ = classify_risk_tier(default_prob)

    xai_sub_tab1, xai_sub_tab2, xai_sub_tab3 = st.tabs([
        "🔍 Local Applicant SHAP Explanation",
        "🍋 LIME Cross-Verification",
        "🌐 Global Feature Importance"
    ])

    with xai_sub_tab1:
        st.markdown("### Local Applicant SHAP Decomposition")
        st.markdown(f"Evaluated with **{active_model_name}** | Assigned Risk: **:{badge_color}[{risk_tier}]** (Default Prob: **{default_prob:.1%}**)")

        shap_results = xai_engine.explain_instance_shap(applicant_proc, top_n=6)
        top_inc = shap_results["top_increasing"]
        top_dec = shap_results["top_decreasing"]

        col_chart, col_narrative = st.columns([1.2, 1])

        with col_chart:
            fig_local = xai_engine.plot_local_contributions(top_inc, top_dec)
            st.pyplot(fig_local)
            plt.close(fig_local)

        with col_narrative:
            st.markdown("#### 📝 Dynamic Decision Narrative")
            st.markdown(shap_results["explanation_text"])

    with xai_sub_tab2:
        st.markdown("### LIME (Local Interpretable Model-agnostic Explanations)")
        st.markdown("LIME trains an interpretable surrogate model locally around the applicant's perturbed inputs to independently corroborate the feature weights.")
        
        lime_exp = xai_engine.explain_instance_lime(applicant_proc, num_features=8)
        
        lime_df = pd.DataFrame(lime_exp, columns=["Rule / Feature Condition", "LIME Weight"])
        lime_df["Impact Type"] = lime_df["LIME Weight"].apply(lambda w: "Risk Elevating" if w > 0 else "Risk Mitigating")
        
        col_lime_tbl, col_lime_chart = st.columns([1, 1.2])
        with col_lime_tbl:
            st.dataframe(lime_df, use_container_width=True)
            st.caption("Positive weights increase estimated default odds; negative weights mitigate default odds.")
        with col_lime_chart:
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#ef4444" if w > 0 else "#10b981" for w in lime_df["LIME Weight"]]
            ax.barh(lime_df["Rule / Feature Condition"], lime_df["LIME Weight"], color=colors)
            ax.axvline(0, color='gray', linestyle='--')
            ax.set_title("LIME Local Surrogate Feature Weights", fontsize=11, fontweight='bold')
            ax.set_xlabel("LIME Weight")
            st.pyplot(fig)
            plt.close(fig)

    with xai_sub_tab3:
        st.markdown("### Global Feature Importance (Bank-Wide Model Ranking)")
        st.markdown("Mean absolute SHAP value across the historical evaluation sample ($N=200$). Highlights the dominant features driving credit decisions across the entire portfolio.")

        fig_global = xai_engine.plot_global_importance(X_test_proc, max_display=14)
        st.pyplot(fig_global)
        plt.close(fig_global)

# -------------------------------------------------------------
# 4. HISTORICAL ANALYSIS
# -------------------------------------------------------------
elif nav_choice == "🏛️ Historical Analysis":
    st.markdown('<div class="main-title">Historical Cohort Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Retrieve actual historical applicants with the most similar financial profiles and inspect their observed ground-truth repayment outcomes.</div>', unsafe_allow_html=True)

    if "applicant_input" not in st.session_state:
        st.warning("Please configure an applicant in the '📝 Applicant Assessment' tab first.")
        st.stop()

    applicant_payload = st.session_state.applicant_input
    applicant_proc = transform_single_applicant(applicant_payload, preprocessor)

    col_k, _ = st.columns([1, 3])
    with col_k:
        k_neighbors = st.slider("Select Historical Cohort Size (Nearest Neighbors)", min_value=3, max_value=15, value=5, step=1)

    # Perform nearest neighbor search
    cohort_results = historical_matcher.find_similar_applicants(applicant_proc, k=k_neighbors)

    # Cohort KPI summary
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Matched Cohort Size</div>
            <div class="kpi-value">{cohort_results['cohort_size']}</div>
            <div class="kpi-sub">Nearest Historical Borrowers</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Observed Repayments</div>
            <div class="kpi-value" style="color: #10b981;">{cohort_results['repaid_count']}</div>
            <div class="kpi-sub">Good Credit Performance</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Observed Defaults</div>
            <div class="kpi-value" style="color: #ef4444;">{cohort_results['default_count']}</div>
            <div class="kpi-sub">Past Delinquent Borrowers</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Cohort Historical Default Rate</div>
            <div class="kpi-value">{cohort_results['cohort_default_rate']:.1%}</div>
            <div class="kpi-sub">Ground Truth in Similar Borrowers</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📋 Comparable Historical Borrower Records")
    st.dataframe(cohort_results["comparison_table"], use_container_width=True)

    st.markdown("""
    > **Target Leakage Safeguard:** The nearest-neighbor algorithm operates solely on independent financial features (credit amount, duration, checking balances, employment, age) without target information. Observed outcomes are derived strictly from historical performance after neighbor identification.
    """)

# -------------------------------------------------------------
# 5. WHAT-IF ANALYSIS
# -------------------------------------------------------------
elif nav_choice == "🔄 What-If Analysis":
    st.markdown('<div class="main-title">Interactive What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Modify applicant financial levers to test risk sensitivity, examine probability shifts, and observe updated XAI drivers.</div>', unsafe_allow_html=True)

    if "applicant_input" not in st.session_state:
        st.warning("Please configure an applicant in the '📝 Applicant Assessment' tab first.")
        st.stop()

    baseline_payload = st.session_state.applicant_input.copy()
    baseline_proc = transform_single_applicant(baseline_payload, preprocessor)
    baseline_prob = float(active_model.predict_proba(baseline_proc)[:, 1][0]) if hasattr(active_model, "predict_proba") else float(active_model.predict(baseline_proc)[0])
    baseline_tier, baseline_color, _ = classify_risk_tier(baseline_prob)

    st.markdown("### 🎛️ Adjust What-If Scenario Levers")

    sim_col1, sim_col2, sim_col3 = st.columns(3)

    with sim_col1:
        mod_credit_amount = st.slider(
            "What-If Credit Amount (DM)",
            min_value=NUMERICAL_SPECS["credit_amount"]["min"],
            max_value=NUMERICAL_SPECS["credit_amount"]["max"],
            value=int(baseline_payload["credit_amount"]),
            step=100
        )
        mod_duration = st.slider(
            "What-If Duration (Months)",
            min_value=NUMERICAL_SPECS["duration"]["min"],
            max_value=NUMERICAL_SPECS["duration"]["max"],
            value=int(baseline_payload["duration"]),
            step=1
        )

    with sim_col2:
        mod_checking = st.selectbox(
            "What-If Checking Status",
            options=CATEGORICAL_OPTIONS["checking_status"],
            index=CATEGORICAL_OPTIONS["checking_status"].index(baseline_payload["checking_status"])
        )
        mod_savings = st.selectbox(
            "What-If Savings Status",
            options=CATEGORICAL_OPTIONS["savings_status"],
            index=CATEGORICAL_OPTIONS["savings_status"].index(baseline_payload["savings_status"])
        )

    with sim_col3:
        mod_guarantor = st.selectbox(
            "What-If Other Debtors / Guarantors",
            options=CATEGORICAL_OPTIONS["other_parties"],
            index=CATEGORICAL_OPTIONS["other_parties"].index(baseline_payload["other_parties"])
        )
        mod_installment = st.selectbox(
            "What-If Installment Rate (% Income)",
            options=[1, 2, 3, 4],
            index=[1, 2, 3, 4].index(int(baseline_payload["installment_rate"]))
        )

    # Build modified scenario payload
    modified_payload = baseline_payload.copy()
    modified_payload.update({
        "credit_amount": mod_credit_amount,
        "duration": mod_duration,
        "checking_status": mod_checking,
        "savings_status": mod_savings,
        "other_parties": mod_guarantor,
        "installment_rate": mod_installment
    })

    mod_proc = transform_single_applicant(modified_payload, preprocessor)
    mod_prob = float(active_model.predict_proba(mod_proc)[:, 1][0]) if hasattr(active_model, "predict_proba") else float(active_model.predict(mod_proc)[0])
    mod_tier, mod_color, _ = classify_risk_tier(mod_prob)

    delta_prob = mod_prob - baseline_prob

    st.markdown("---")
    st.markdown("### 📊 Scenario Comparison: Baseline vs What-If")

    comp_c1, comp_c2, comp_c3 = st.columns(3)

    with comp_c1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Original Baseline</div>
            <div class="kpi-value" style="color: {baseline_color};">{baseline_tier}</div>
            <div class="kpi-sub">Default Probability: <strong>{baseline_prob:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">What-If Simulation</div>
            <div class="kpi-value" style="color: {mod_color};">{mod_tier}</div>
            <div class="kpi-sub">Default Probability: <strong>{mod_prob:.1%}</strong></div>
        </div>
        """, unsafe_allow_html=True)

    with comp_c3:
        delta_color = "#10b981" if delta_prob < 0 else "#ef4444"
        delta_sign = "+" if delta_prob > 0 else ""
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Net Risk Shift (Δ P)</div>
            <div class="kpi-value" style="color: {delta_color};">{delta_sign}{delta_prob:.1%}</div>
            <div class="kpi-sub">{'Risk Reduced' if delta_prob < 0 else ('Risk Increased' if delta_prob > 0 else 'No Change')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Narrative explanation of change
    st.markdown("#### 💡 What Caused the Prediction Shift?")
    reasons = []
    if mod_credit_amount != baseline_payload["credit_amount"]:
        reasons.append(f"Adjusting loan principal from {baseline_payload['credit_amount']} DM to {mod_credit_amount} DM")
    if mod_duration != baseline_payload["duration"]:
        reasons.append(f"Changing loan tenure from {baseline_payload['duration']} months to {mod_duration} months")
    if mod_checking != baseline_payload["checking_status"]:
        reasons.append(f"Altering checking account liquidity tier from '{baseline_payload['checking_status']}' to '{mod_checking}'")
    if mod_savings != baseline_payload["savings_status"]:
        reasons.append(f"Modifying savings reserves tier from '{baseline_payload['savings_status']}' to '{mod_savings}'")
    if mod_guarantor != baseline_payload["other_parties"]:
        reasons.append(f"Adding/modifying credit guarantor status to '{mod_guarantor}'")

    if reasons:
        explanation_msg = (
            f"The net change of **{delta_sign}{delta_prob:.1%}** in default probability was driven by: " +
            ", ".join(reasons) + ". "
            "In historical lending data, shorter maturities and higher borrower liquidity buffer against delinquency. "
            "*Important: Statistical associations in model predictions do not constitute guaranteed causal effects.*"
        )
    else:
        explanation_msg = "No input parameters were changed. Baseline and What-If scenarios are identical."

    st.info(explanation_msg)

# -------------------------------------------------------------
# 6. MODEL COMPARISON
# -------------------------------------------------------------
elif nav_choice == "⚖️ Model Comparison":
    st.markdown('<div class="main-title">Machine Learning Model Comparison & Governance</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Evaluation of Logistic Regression, Decision Tree, Random Forest, and Support Vector Machine on out-of-sample test split.</div>', unsafe_allow_html=True)

    metrics_df = pd.DataFrame(metrics_summary).T
    metrics_df = metrics_df[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]

    st.markdown("#### 🏆 Performance Metrics Summary (Test Split $N=200$)")
    st.dataframe(metrics_df.style.highlight_max(axis=0, color="#dcfce7"), use_container_width=True)

    st.markdown("""
    > **Selection Rationale:** In lending credit risk management, **Recall** (identifying true defaulters) and **ROC-AUC** (discriminative separation) are the paramount criteria. Misclassifying an actual defaulter as creditworthy incurs five times higher losses than a false denial. **Random Forest** provides an optimal synergy of high discriminative power (ROC-AUC ~0.80), balanced recall, and exact tree-based SHAP explainability.
    """)

    # Interactive Confusion Matrices
    st.markdown("---")
    st.markdown("#### 🔲 Confusion Matrices Across All 4 Models")

    cm_cols = st.columns(4)
    for idx, (m_name, cm_vals) in enumerate(confusion_matrices.items()):
        with cm_cols[idx]:
            st.markdown(f"**{m_name}**")
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(
                cm_vals,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                xticklabels=["Repaid", "Default"],
                yticklabels=["Repaid", "Default"],
                ax=ax
            )
            ax.set_xlabel("Predicted Class", fontsize=8)
            ax.set_ylabel("Actual Class", fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

    # ROC Curves Overlay
    st.markdown("---")
    st.markdown("#### 📈 ROC Curves Overlay")
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    for m_name, curve_info in roc_curves_data.items():
        ax.plot(curve_info["fpr"], curve_info["tpr"], label=f"{m_name} (AUC = {curve_info['auc']:.3f})")
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Random Guess (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10)
    ax.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=10)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison", fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)

# -------------------------------------------------------------
# 7. RESPONSIBLE AI & FAIRNESS AUDIT
# -------------------------------------------------------------
elif nav_choice == "🛡️ Responsible AI & Fairness":
    st.markdown('<div class="main-title">Responsible AI & Fair Lending Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Quantitative parity audits, disparate impact assessments, and regulatory compliance disclosures.</div>', unsafe_allow_html=True)

    y_test_pred = active_model.predict(X_test_proc)
    fairness_eval = FairnessEvaluator(
        df=train_data_bundle["X_test"],
        y_pred=y_test_pred,
        y_true=y_test
    )

    gender_res = fairness_eval.evaluate_gender_fairness()
    age_res = fairness_eval.evaluate_age_fairness(age_cutoff=25)
    foreign_res = fairness_eval.evaluate_foreign_worker_fairness()

    f_col1, f_col2, f_col3 = st.columns(3)

    with f_col1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Gender Disparate Impact</div>
            <div class="kpi-value">{gender_res['disparate_impact_ratio']:.2f}</div>
            <div class="kpi-sub">Female: {gender_res['female_approval_rate']:.1%} | Male: {gender_res['male_approval_rate']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with f_col2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Age Disparate Impact (< 25)</div>
            <div class="kpi-value">{age_res['disparate_impact_ratio']:.2f}</div>
            <div class="kpi-sub">Young: {age_res['young_approval_rate']:.1%} | Mature: {age_res['mature_approval_rate']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    with f_col3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">Foreign Worker Ratio</div>
            <div class="kpi-value">{foreign_res['disparate_impact_ratio']:.2f}</div>
            <div class="kpi-sub">Foreign: {foreign_res['foreign_approval_rate']:.1%} | Domestic: {foreign_res['domestic_approval_rate']:.1%}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### ⚖️ Regulatory Governance Benchmarks")
    st.markdown("""
    - **Four-Fifths (80%) Rule:** Under US EEOC and Fair Lending Guidelines, a selection rate for any group which is less than 80% (0.80) of the rate for the group with the highest rate will generally be regarded as evidence of adverse impact.
    - **Historical Bias Alert:** The German Credit dataset reflects lending practices and demographic distributions from historical German bank cohorts. Societal and historical biases present in training data must be scrutinized and mitigated before production lending authorization.
    - **Fair Lending Obligation:** Sensitive attributes (such as gender and foreign worker status) should not serve as unilateral criteria for credit denial. Human underwriter oversight is mandated.
    """)

# -------------------------------------------------------------
# 8. PREDICTION HISTORY
# -------------------------------------------------------------
elif nav_choice == "📜 Prediction History":
    st.markdown('<div class="main-title">Assessment Audit Trail & History</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Session assessment log with anonymized applicant identifiers and privacy protections.</div>', unsafe_allow_html=True)

    history_df = history_store.get_history_dataframe()

    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)

        col_dl, col_clr, _ = st.columns([1, 1, 2])
        with col_dl:
            csv_data = history_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Audit Log (CSV)",
                data=csv_data,
                file_name="credit_assessment_audit_log.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col_clr:
            if st.button("🗑️ Clear History", use_container_width=True):
                history_store.clear_history()
                st.rerun()
    else:
        st.info("No assessments logged in this session yet. Evaluate an applicant in the '📝 Applicant Assessment' tab to create an audit record.")
