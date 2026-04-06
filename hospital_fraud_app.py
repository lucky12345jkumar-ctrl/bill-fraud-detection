# (imports unchanged)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import io
import base64
from datetime import datetime

# PAGE CONFIG
st.set_page_config(
    page_title="MedBillGuard – Hospital Bill Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS (unchanged)
st.markdown("""<style>/* your CSS unchanged */</style>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & TRAIN
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("billing_fraud_dataset_rows_csv.csv")

    df["BillingRatio"] = df["BillingAmount"] / (df["ApprovedAmount"] + 1)
    df["CostPerProcedure"] = df["BillingAmount"] / (df["NumProcedures"] + 1)
    df["CostPerDay"] = df["BillingAmount"] / (df["TreatmentDurationDays"] + 1)
    df["OverchargeAmount"] = df["BillingAmount"] - df["ApprovedAmount"]
    df["OverchargePct"] = df["OverchargeAmount"] / (df["ApprovedAmount"] + 1) * 100

    le = LabelEncoder()
    df["TreatmentTypeEnc"] = le.fit_transform(df["TreatmentType"])

    features = [
        "TreatmentDurationDays","BillingAmount","ApprovedAmount",
        "NumProcedures","BillingRatio","CostPerProcedure",
        "CostPerDay","OverchargeAmount","OverchargePct","TreatmentTypeEnc"
    ]

    X = df[features]
    y = df["FraudFlag"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)

    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_scaled)

    stats = df.groupby("TreatmentType").agg(
        billing_mean=("BillingAmount","mean"),
        billing_std=("BillingAmount","std"),
        duration_mean=("TreatmentDurationDays","mean"),
        duration_std=("TreatmentDurationDays","std"),
        proc_mean=("NumProcedures","mean"),
        proc_std=("NumProcedures","std"),
    ).fillna(0)

    return df, rf, iso, scaler, le, features, stats, X_test, y_test


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────
def score_color(score):
    if score >= 60: return "#ef4444"
    if score >= 35: return "#f59e0b"
    return "#22c55e"


def predict_fraud(input_dict, rf, iso, scaler, le, features, stats):
    billing = input_dict["BillingAmount"]
    approved = input_dict["ApprovedAmount"]
    duration = input_dict["TreatmentDurationDays"]
    num_proc = input_dict["NumProcedures"]

    billing_ratio = billing / (approved + 1)
    overcharge_pct = (billing - approved) / (approved + 1) * 100

    row = [[duration, billing, approved, num_proc,
            billing_ratio, billing/(num_proc+1),
            billing/(duration+1), billing-approved,
            overcharge_pct, 0]]

    row_scaled = scaler.transform(row)

    rf_prob = rf.predict_proba(row_scaled)[0][1]
    iso_flag = iso.predict(row_scaled)[0]

    score = rf_prob * 100

    if score >= 60:
        verdict = "FRAUD"
    elif score >= 35:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGITIMATE"

    return {"verdict": verdict, "score": score}


def bulk_predict(df, rf, iso, scaler, le, features, stats):
    results = []
    for _, row in df.iterrows():
        res = predict_fraud(row, rf, iso, scaler, le, features, stats)
        results.append({
            "ClaimID": row["ClaimID"],
            "FraudScore(%)": round(res["score"],1),
            "Verdict": res["verdict"]
        })
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
df, rf, iso, scaler, le, features, stats, X_test, y_test = load_and_train()

# ─────────────────────────────────────────────
# BULK SCANNER PAGE (FIXED)
# ─────────────────────────────────────────────
st.title("📁 Bulk Scanner")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

if uploaded:
    df_upload = pd.read_csv(uploaded)

    results_df = bulk_predict(df_upload, rf, iso, scaler, le, features, stats)

    st.write("### Results")

    display_df = results_df.copy()

    # ✅ FIXED INDENTATION BELOW
    def color_verdict(val):
        if val == "FRAUD":
            return "background-color:#7f1d1d; color:#fca5a5"
        if val == "SUSPICIOUS":
            return "background-color:#78350f; color:#fcd34d"
        return "background-color:#14532d; color:#86efac"

    def color_score(val):
        return f"color:{score_color(val)}; font-weight:700"

    styled = (
        display_df.style
        .map(color_verdict, subset=["Verdict"])
        .map(color_score, subset=["FraudScore(%)"])
    )

    st.dataframe(styled, use_container_width=True)
