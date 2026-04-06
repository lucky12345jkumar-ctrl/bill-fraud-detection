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

# ─────────────────────────────────────────────
#  Page config & custom CSS
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="MedBillGuard – Hospital Bill Fraud Detection",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] { background: #0d1117; }
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
* { font-family: 'Inter', sans-serif; }

/* ── Metric cards ── */
.metric-card {
    background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 20px 24px;
    margin: 6px 0;
}
.metric-title { color: #9ca3af; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }
.metric-value { color: #f9fafb; font-size: 2rem; font-weight: 700; line-height: 1; }
.metric-sub   { color: #6b7280; font-size: 0.75rem; margin-top: 4px; }

/* ── Fraud badge ── */
.badge-fraud    { background:#7f1d1d; color:#fca5a5; border:1px solid #991b1b; border-radius:6px; padding:4px 12px; font-weight:700; font-size:0.85rem; }
.badge-legit    { background:#14532d; color:#86efac; border:1px solid #166534; border-radius:6px; padding:4px 12px; font-weight:700; font-size:0.85rem; }
.badge-suspicious { background:#78350f; color:#fcd34d; border:1px solid #92400e; border-radius:6px; padding:4px 12px; font-weight:700; font-size:0.85rem; }

/* ── Result box ── */
.result-box {
    border-radius: 12px;
    padding: 24px;
    margin-top: 16px;
    border: 1px solid;
}
.result-fraud { background:#1c0a0a; border-color:#7f1d1d; }
.result-legit { background:#0a1c0f; border-color:#14532d; }
.result-suspicious { background:#1c1200; border-color:#78350f; }

/* ── Section header ── */
.section-header {
    font-size: 1.1rem;
    font-weight: 700;
    color: #e2e8f0;
    border-left: 4px solid #3b82f6;
    padding-left: 12px;
    margin: 20px 0 12px;
}

/* ── Risk bar ── */
.risk-bar-outer { background:#1f2937; border-radius:999px; height:12px; margin:6px 0; }
.risk-bar-inner { border-radius:999px; height:12px; transition: width 0.5s ease; }

/* ── Tab styling ── */
[data-baseweb="tab-list"] { background:#161b22; border-radius:8px; gap:4px; padding:4px; }
[data-baseweb="tab"]      { color:#9ca3af !important; border-radius:6px !important; }
[aria-selected="true"]    { background:#1d4ed8 !important; color:#fff !important; }

/* ── Button ── */
.stButton > button { background: linear-gradient(135deg,#1d4ed8,#7c3aed); color:#fff; border:none; border-radius:8px; font-weight:600; padding:10px 28px; }
.stButton > button:hover { opacity:0.88; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load & train model (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_and_train():
    """Load dataset and train a RandomForest + IsolationForest ensemble."""
    df = pd.read_csv("billing_fraud_dataset_rows_csv.csv")

    # Feature engineering
    df["BillingRatio"]       = df["BillingAmount"] / (df["ApprovedAmount"] + 1)
    df["CostPerProcedure"]   = df["BillingAmount"] / (df["NumProcedures"] + 1)
    df["CostPerDay"]         = df["BillingAmount"] / (df["TreatmentDurationDays"] + 1)
    df["OverchargeAmount"]   = df["BillingAmount"] - df["ApprovedAmount"]
    df["OverchargePct"]      = df["OverchargeAmount"] / (df["ApprovedAmount"] + 1) * 100

    le = LabelEncoder()
    df["TreatmentTypeEnc"] = le.fit_transform(df["TreatmentType"])

    features = [
        "TreatmentDurationDays", "BillingAmount", "ApprovedAmount",
        "NumProcedures", "BillingRatio", "CostPerProcedure",
        "CostPerDay", "OverchargeAmount", "OverchargePct", "TreatmentTypeEnc"
    ]
    X = df[features]
    y = df["FraudFlag"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Supervised model
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)

    # Unsupervised anomaly model
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_scaled)

    # Per-treatment stats for rule-based checks
    stats = df.groupby("TreatmentType").agg(
        billing_mean=("BillingAmount", "mean"),
        billing_std=("BillingAmount", "std"),
        duration_mean=("TreatmentDurationDays", "mean"),
        duration_std=("TreatmentDurationDays", "std"),
        proc_mean=("NumProcedures", "mean"),
        proc_std=("NumProcedures", "std"),
    ).fillna(0)

    return df, rf, iso, scaler, le, features, stats, X_test, y_test


@st.cache_data
def get_dataset_stats(df):
    total        = len(df)
    fraud_count  = df["FraudFlag"].sum()
    fraud_pct    = fraud_count / total * 100
    avg_overcharge = df["OverchargeAmount"].mean() if "OverchargeAmount" in df.columns else (
        df["BillingAmount"] - df["ApprovedAmount"]).mean()
    return total, fraud_count, fraud_pct, avg_overcharge


# ─────────────────────────────────────────────
#  Prediction helper
# ─────────────────────────────────────────────
def predict_fraud(input_dict, rf, iso, scaler, le, features, stats):
    tt = input_dict["TreatmentType"]
    billing    = input_dict["BillingAmount"]
    approved   = input_dict["ApprovedAmount"]
    duration   = input_dict["TreatmentDurationDays"]
    num_proc   = input_dict["NumProcedures"]

    billing_ratio     = billing / (approved + 1)
    cost_per_proc     = billing / (num_proc + 1)
    cost_per_day      = billing / (duration + 1)
    overcharge_amt    = billing - approved
    overcharge_pct    = overcharge_amt / (approved + 1) * 100
    tt_enc            = le.transform([tt])[0] if tt in le.classes_ else 0

    row = [[duration, billing, approved, num_proc,
            billing_ratio, cost_per_proc, cost_per_day,
            overcharge_amt, overcharge_pct, tt_enc]]
    row_scaled = scaler.transform(row)

    rf_prob   = rf.predict_proba(row_scaled)[0][1]
    iso_score = iso.decision_function(row_scaled)[0]   # more negative = more anomalous
    iso_flag  = iso.predict(row_scaled)[0]             # -1 = anomaly

    # Rule-based flags
    rule_flags = []
    if tt in stats.index:
        s = stats.loc[tt]
        if billing > s["billing_mean"] + 2.5 * s["billing_std"]:
            rule_flags.append("Billing significantly above treatment average")
        if duration > s["duration_mean"] + 2.5 * s["duration_std"]:
            rule_flags.append("Treatment duration unusually long")
        if num_proc > s["proc_mean"] + 2.5 * s["proc_std"]:
            rule_flags.append("Abnormally high number of procedures")
    if billing_ratio > 1.5:
        rule_flags.append(f"Billing is {billing_ratio:.1f}× the approved amount")
    if overcharge_pct > 80:
        rule_flags.append(f"Overcharge is {overcharge_pct:.0f}% of approved amount")
    if iso_flag == -1:
        rule_flags.append("Statistical outlier detected (Isolation Forest)")

    # Ensemble score
    iso_contrib = max(0, min(1, (-iso_score + 0.2) / 0.5))
    rule_contrib = min(1.0, len(rule_flags) * 0.15)
    ensemble = 0.55 * rf_prob + 0.25 * iso_contrib + 0.20 * rule_contrib

    if ensemble >= 0.60:
        verdict = "FRAUD"
    elif ensemble >= 0.35:
        verdict = "SUSPICIOUS"
    else:
        verdict = "LEGITIMATE"

    return {
        "verdict": verdict,
        "ensemble_score": ensemble,
        "rf_prob": rf_prob,
        "iso_score": iso_score,
        "iso_flag": iso_flag,
        "rule_flags": rule_flags,
        "billing_ratio": billing_ratio,
        "overcharge_pct": overcharge_pct,
        "cost_per_day": cost_per_day,
        "cost_per_proc": cost_per_proc,
    }


def bulk_predict(df_in, rf, iso, scaler, le, features, stats):
    results = []
    for _, row in df_in.iterrows():
        inp = {
            "TreatmentType": row.get("TreatmentType", "Consultation"),
            "BillingAmount": float(row.get("BillingAmount", 0)),
            "ApprovedAmount": float(row.get("ApprovedAmount", 0)),
            "TreatmentDurationDays": float(row.get("TreatmentDurationDays", 1)),
            "NumProcedures": float(row.get("NumProcedures", 1)),
        }
        res = predict_fraud(inp, rf, iso, scaler, le, features, stats)
        results.append({
            "ClaimID":            row.get("ClaimID", "—"),
            "PatientID":          row.get("PatientID", "—"),
            "ProviderID":         row.get("ProviderID", "—"),
            "TreatmentType":      inp["TreatmentType"],
            "BillingAmount":      inp["BillingAmount"],
            "ApprovedAmount":     inp["ApprovedAmount"],
            "Duration(Days)":     inp["TreatmentDurationDays"],
            "NumProcedures":      inp["NumProcedures"],
            "FraudScore(%)":      round(res["ensemble_score"] * 100, 1),
            "Verdict":            res["verdict"],
            "RuleFlags":          "; ".join(res["rule_flags"]) if res["rule_flags"] else "—",
        })
    return pd.DataFrame(results)


def score_color(score):
    if score >= 60: return "#ef4444"
    if score >= 35: return "#f59e0b"
    return "#22c55e"


def verdict_badge(v):
    if v == "FRAUD":      return '<span class="badge-fraud">🚨 FRAUD</span>'
    if v == "SUSPICIOUS": return '<span class="badge-suspicious">⚠️ SUSPICIOUS</span>'
    return '<span class="badge-legit">✅ LEGITIMATE</span>'


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 MedBillGuard")
    st.markdown("**Hospital Bill Fraud Detection System**")
    st.divider()
    st.markdown("### Navigation")
    page = st.radio("", [
        "📊 Dashboard",
        "🔍 Manual Prediction",
        "📁 Bulk Scanner",
        "📈 Model Insights",
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("### About")
    st.markdown("""
    This system detects fraudulent or inflated hospital bills using:
    - 🌲 **Random Forest** classifier
    - 🎯 **Isolation Forest** anomaly detection
    - 📏 **Rule-based** heuristics
    - 🔗 **Ensemble** scoring
    """)
    st.divider()
    st.markdown("<small style='color:#6b7280'>Model trained on 500 hospital claims</small>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────
with st.spinner("Loading model…"):
    df, rf, iso, scaler, le, features, stats, X_test, y_test = load_and_train()

# Derived columns for display
df["BillingRatio"]     = df["BillingAmount"] / (df["ApprovedAmount"] + 1)
df["OverchargeAmount"] = df["BillingAmount"] - df["ApprovedAmount"]
df["OverchargePct"]    = df["OverchargeAmount"] / (df["ApprovedAmount"] + 1) * 100
df["CostPerDay"]       = df["BillingAmount"] / (df["TreatmentDurationDays"] + 1)
df["CostPerProcedure"] = df["BillingAmount"] / (df["NumProcedures"] + 1)
total, fraud_count, fraud_pct, avg_overcharge = get_dataset_stats(df)


# ═══════════════════════════════════════════════════
#  PAGE: Dashboard
# ═══════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 📊 Fraud Detection Dashboard")
    st.markdown("Real-time analytics on hospital billing patterns and fraud indicators.")
    st.divider()

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Total Claims</div>
            <div class="metric-value">{total:,}</div>
            <div class="metric-sub">In dataset</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Fraud Cases</div>
            <div class="metric-value" style="color:#f87171">{int(fraud_count)}</div>
            <div class="metric-sub">{fraud_pct:.1f}% of all claims</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Avg Overcharge</div>
            <div class="metric-value" style="color:#fbbf24">₹{avg_overcharge:,.0f}</div>
            <div class="metric-sub">Per fraudulent claim</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        legit = total - int(fraud_count)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Legitimate Claims</div>
            <div class="metric-value" style="color:#4ade80">{legit:,}</div>
            <div class="metric-sub">{100-fraud_pct:.1f}% of all claims</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns(2)

    with c_left:
        # Fraud by treatment type
        fraud_by_type = df.groupby("TreatmentType")["FraudFlag"].agg(["sum","count"]).reset_index()
        fraud_by_type["FraudRate"] = fraud_by_type["sum"] / fraud_by_type["count"] * 100
        fig = px.bar(fraud_by_type, x="TreatmentType", y="FraudRate",
                     color="FraudRate",
                     color_continuous_scale=["#22c55e","#f59e0b","#ef4444"],
                     title="Fraud Rate by Treatment Type (%)",
                     labels={"FraudRate":"Fraud Rate (%)","TreatmentType":"Treatment"})
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", title_font_color="#e2e8f0",
                          coloraxis_showscale=False)
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        # Billing vs Approved scatter
        fig2 = px.scatter(df, x="ApprovedAmount", y="BillingAmount",
                          color=df["FraudFlag"].map({0:"Legitimate",1:"Fraud"}),
                          color_discrete_map={"Fraud":"#ef4444","Legitimate":"#3b82f6"},
                          title="Billing vs Approved Amount",
                          labels={"ApprovedAmount":"Approved (₹)","BillingAmount":"Billed (₹)"},
                          opacity=0.7)
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", title_font_color="#e2e8f0")
        st.plotly_chart(fig2, use_container_width=True)

    c_left2, c_right2 = st.columns(2)
    with c_left2:
        # Overcharge distribution
        fig3 = px.histogram(df, x="OverchargePct", color=df["FraudFlag"].map({0:"Legitimate",1:"Fraud"}),
                            color_discrete_map={"Fraud":"#ef4444","Legitimate":"#3b82f6"},
                            nbins=40, title="Overcharge % Distribution",
                            labels={"OverchargePct":"Overcharge (%)"})
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", title_font_color="#e2e8f0", barmode="overlay")
        fig3.update_traces(opacity=0.75)
        st.plotly_chart(fig3, use_container_width=True)

    with c_right2:
        # Treatment duration vs billing amount
        fig4 = px.scatter(df, x="TreatmentDurationDays", y="CostPerDay",
                          color=df["FraudFlag"].map({0:"Legitimate",1:"Fraud"}),
                          color_discrete_map={"Fraud":"#ef4444","Legitimate":"#3b82f6"},
                          title="Duration vs Daily Cost",
                          labels={"TreatmentDurationDays":"Duration (Days)","CostPerDay":"Cost/Day (₹)"},
                          size="NumProcedures", size_max=20, opacity=0.7)
        fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0", title_font_color="#e2e8f0")
        st.plotly_chart(fig4, use_container_width=True)

    # Recent flagged claims
    st.markdown('<div class="section-header">Recent High-Risk Claims (Top 10 by Overcharge)</div>', unsafe_allow_html=True)
    top_risk = df[df["FraudFlag"] == 1].nlargest(10, "OverchargeAmount")[
        ["ClaimID","PatientID","ProviderID","TreatmentType","BillingAmount","ApprovedAmount","OverchargeAmount","OverchargePct"]
    ].copy()
    top_risk.columns = ["Claim ID","Patient ID","Provider ID","Treatment","Billed (₹)","Approved (₹)","Overcharge (₹)","Overcharge %"]
    st.dataframe(top_risk.style.format({
        "Billed (₹)":"₹{:,.0f}","Approved (₹)":"₹{:,.0f}","Overcharge (₹)":"₹{:,.0f}","Overcharge %":"{:.1f}%"
    }).highlight_max(subset=["Overcharge (₹)"], color="#7f1d1d"), use_container_width=True)


# ═══════════════════════════════════════════════════
#  PAGE: Manual Prediction
# ═══════════════════════════════════════════════════
elif page == "🔍 Manual Prediction":
    st.markdown("# 🔍 Manual Claim Prediction")
    st.markdown("Enter claim details below to assess fraud risk in real time.")
    st.divider()

    with st.form("prediction_form"):
        st.markdown('<div class="section-header">Claim Information</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            claim_id   = st.text_input("Claim ID", value="C99999")
            patient_id = st.text_input("Patient ID", value="P001")
        with col2:
            provider_id   = st.text_input("Provider ID", value="D100")
            treatment_type = st.selectbox("Treatment Type", ["Consultation","Surgery","Therapy","Medication"])
        with col3:
            duration   = st.number_input("Treatment Duration (Days)", min_value=1, max_value=120, value=5)
            num_proc   = st.number_input("Number of Procedures", min_value=1, max_value=50, value=3)

        st.markdown('<div class="section-header">Billing Details</div>', unsafe_allow_html=True)
        col4, col5 = st.columns(2)
        with col4:
            billing_amount  = st.number_input("Billing Amount (₹)", min_value=100, max_value=1_000_000, value=25000, step=500)
        with col5:
            approved_amount = st.number_input("Approved Amount (₹)", min_value=100, max_value=1_000_000, value=20000, step=500)

        submitted = st.form_submit_button("🔍 Analyze Claim", use_container_width=True)

    if submitted:
        inp = {
            "TreatmentType": treatment_type,
            "BillingAmount": float(billing_amount),
            "ApprovedAmount": float(approved_amount),
            "TreatmentDurationDays": float(duration),
            "NumProcedures": float(num_proc),
        }
        result = predict_fraud(inp, rf, iso, scaler, le, features, stats)
        verdict = result["verdict"]
        score_pct = result["ensemble_score"] * 100

        # Pick CSS class
        box_cls = {"FRAUD": "result-fraud", "SUSPICIOUS": "result-suspicious", "LEGITIMATE": "result-legit"}[verdict]

        # ── Result header ──
        st.markdown(f"""<div class="result-box {box_cls}">
            <div style="display:flex; justify-content:space-between; align-items:center">
                <div>
                    <div style="color:#9ca3af; font-size:0.8rem; margin-bottom:6px">VERDICT FOR CLAIM {claim_id}</div>
                    {verdict_badge(verdict)}
                </div>
                <div style="text-align:right">
                    <div style="color:#9ca3af; font-size:0.8rem">FRAUD RISK SCORE</div>
                    <div style="font-size:2.5rem; font-weight:800; color:{score_color(score_pct)}">{score_pct:.1f}%</div>
                </div>
            </div>
            <div style="margin-top:16px">
                <div style="color:#9ca3af; font-size:0.75rem; margin-bottom:4px">Risk Level</div>
                <div class="risk-bar-outer">
                    <div class="risk-bar-inner" style="width:{min(score_pct,100):.1f}%; background:{score_color(score_pct)}"></div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Metric breakdown ──
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">RF Model Score</div>
                <div class="metric-value" style="font-size:1.6rem; color:{score_color(result['rf_prob']*100)}">{result['rf_prob']*100:.1f}%</div>
                <div class="metric-sub">Random Forest probability</div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Billing Ratio</div>
                <div class="metric-value" style="font-size:1.6rem; color:{'#ef4444' if result['billing_ratio']>1.5 else '#4ade80'}">{result['billing_ratio']:.2f}×</div>
                <div class="metric-sub">Billed ÷ Approved</div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Overcharge %</div>
                <div class="metric-value" style="font-size:1.6rem; color:{'#ef4444' if result['overcharge_pct']>50 else '#4ade80'}">{result['overcharge_pct']:.1f}%</div>
                <div class="metric-sub">Above approved amount</div>
            </div>""", unsafe_allow_html=True)
        with col_d:
            anomaly_label = "⚠️ Anomaly" if result["iso_flag"] == -1 else "✅ Normal"
            anomaly_color = "#f87171" if result["iso_flag"] == -1 else "#4ade80"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Anomaly Check</div>
                <div class="metric-value" style="font-size:1.2rem; color:{anomaly_color}">{anomaly_label}</div>
                <div class="metric-sub">Isolation Forest result</div>
            </div>""", unsafe_allow_html=True)

        # ── Rule-based flags ──
        st.markdown('<div class="section-header">Rule-Based Audit Flags</div>', unsafe_allow_html=True)
        if result["rule_flags"]:
            for flag in result["rule_flags"]:
                st.error(f"🚩 {flag}")
        else:
            st.success("✅ No rule violations detected.")

        # ── Contextual stats ──
        st.markdown('<div class="section-header">Benchmark vs Treatment Averages</div>', unsafe_allow_html=True)
        if treatment_type in stats.index:
            s = stats.loc[treatment_type]
            bench_df = pd.DataFrame({
                "Metric":        ["Billing Amount (₹)", "Duration (Days)", "Procedures"],
                "This Claim":    [billing_amount, duration, num_proc],
                "Category Avg":  [round(s["billing_mean"]), round(s["duration_mean"]), round(s["proc_mean"])],
                "Category Std":  [round(s["billing_std"]), round(s["duration_std"]), round(s["proc_std"])],
            })
            bench_df["Deviation (σ)"] = [
                round((billing_amount - s["billing_mean"]) / (s["billing_std"] + 1), 2),
                round((duration - s["duration_mean"]) / (s["duration_std"] + 1), 2),
                round((num_proc - s["proc_mean"]) / (s["proc_std"] + 1), 2),
            ]
            st.dataframe(bench_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════
#  PAGE: Bulk Scanner
# ═══════════════════════════════════════════════════
elif page == "📁 Bulk Scanner":
    st.markdown("# 📁 Bulk Claim Scanner")
    st.markdown("Upload a CSV file of claims to scan for fraud in batch.")
    st.divider()

    # Template download
    template_df = pd.DataFrame({
        "ClaimID":             ["C10001","C10002"],
        "PatientID":           ["P001","P002"],
        "ProviderID":          ["D100","D200"],
        "TreatmentType":       ["Surgery","Consultation"],
        "TreatmentDurationDays": [10, 2],
        "BillingAmount":       [150000, 5000],
        "ApprovedAmount":      [80000, 4500],
        "NumProcedures":       [5, 1],
    })
    csv_bytes = template_df.to_csv(index=False).encode()
    st.download_button("⬇️ Download CSV Template", csv_bytes, "claim_template.csv", "text/csv")

    st.markdown("<br>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload Claims CSV", type=["csv"],
                                 help="Must have columns: ClaimID, PatientID, ProviderID, TreatmentType, TreatmentDurationDays, BillingAmount, ApprovedAmount, NumProcedures")

    # Also offer to scan the built-in dataset
    use_builtin = st.checkbox("Or scan the built-in dataset (500 claims)")

    if uploaded or use_builtin:
        if uploaded:
            df_upload = pd.read_csv(uploaded)
        else:
            df_upload = df.copy()

        required_cols = {"TreatmentType","BillingAmount","ApprovedAmount","TreatmentDurationDays","NumProcedures"}
        missing = required_cols - set(df_upload.columns)
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        with st.spinner(f"Scanning {len(df_upload):,} claims…"):
            results_df = bulk_predict(df_upload, rf, iso, scaler, le, features, stats)

        # ── Summary cards ──
        st.markdown('<div class="section-header">Scan Results Summary</div>', unsafe_allow_html=True)
        total_scanned = len(results_df)
        fraud_cnt     = (results_df["Verdict"] == "FRAUD").sum()
        susp_cnt      = (results_df["Verdict"] == "SUSPICIOUS").sum()
        legit_cnt     = (results_df["Verdict"] == "LEGITIMATE").sum()
        avg_score     = results_df["FraudScore(%)"].mean()

        mc1, mc2, mc3, mc4 = st.columns(4)
        with mc1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Total Scanned</div>
                <div class="metric-value">{total_scanned:,}</div>
            </div>""", unsafe_allow_html=True)
        with mc2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Flagged as Fraud</div>
                <div class="metric-value" style="color:#f87171">{fraud_cnt}</div>
                <div class="metric-sub">{fraud_cnt/total_scanned*100:.1f}% of claims</div>
            </div>""", unsafe_allow_html=True)
        with mc3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Suspicious</div>
                <div class="metric-value" style="color:#fbbf24">{susp_cnt}</div>
                <div class="metric-sub">{susp_cnt/total_scanned*100:.1f}% of claims</div>
            </div>""", unsafe_allow_html=True)
        with mc4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-title">Avg Fraud Score</div>
                <div class="metric-value" style="color:{score_color(avg_score)}">{avg_score:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # ── Charts ──
        st.markdown("<br>", unsafe_allow_html=True)
        ch1, ch2 = st.columns(2)
        with ch1:
            pie_data = pd.DataFrame({"Verdict":["Fraud","Suspicious","Legitimate"],"Count":[fraud_cnt,susp_cnt,legit_cnt]})
            fig_pie = px.pie(pie_data, names="Verdict", values="Count",
                             color="Verdict",
                             color_discrete_map={"Fraud":"#ef4444","Suspicious":"#f59e0b","Legitimate":"#22c55e"},
                             title="Verdict Distribution")
            fig_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0", title_font_color="#e2e8f0")
            st.plotly_chart(fig_pie, use_container_width=True)

        with ch2:
            fig_hist = px.histogram(results_df, x="FraudScore(%)", color="Verdict",
                                    color_discrete_map={"FRAUD":"#ef4444","SUSPICIOUS":"#f59e0b","LEGITIMATE":"#22c55e"},
                                    nbins=30, title="Fraud Score Distribution",
                                    labels={"FraudScore(%)":"Fraud Score (%)"})
            fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                   font_color="#e2e8f0", title_font_color="#e2e8f0")
            st.plotly_chart(fig_hist, use_container_width=True)

        # ── Filters & Table ──
        st.markdown('<div class="section-header">Detailed Results</div>', unsafe_allow_html=True)
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            filter_verdict = st.multiselect("Filter by Verdict", ["FRAUD","SUSPICIOUS","LEGITIMATE"],
                                             default=["FRAUD","SUSPICIOUS"])
        with col_f2:
            min_score = st.slider("Minimum Fraud Score (%)", 0, 100, 0)
        with col_f3:
            filter_treatment = st.multiselect("Filter by Treatment", df["TreatmentType"].unique().tolist(),
                                               default=df["TreatmentType"].unique().tolist())

        display_df = results_df[
            (results_df["Verdict"].isin(filter_verdict)) &
            (results_df["FraudScore(%)"] >= min_score) &
            (results_df["TreatmentType"].isin(filter_treatment))
        ].sort_values("FraudScore(%)", ascending=False)

        st.info(f"Showing **{len(display_df)}** claims matching filters.")

        def color_verdict(val):
            if val == "FRAUD":      return "background-color:#7f1d1d; color:#fca5a5"
            if val == "SUSPICIOUS": return "background-color:#78350f; color:#fcd34d"
            return "background-color:#14532d; color:#86efac"

        def color_score(val):
            c = score_color(val)
            return f"color:{c}; font-weight:700"

        styled = display_df.style\
            .applymap(color_verdict, subset=["Verdict"])\
            .applymap(color_score, subset=["FraudScore(%)"])

        st.dataframe(styled, use_container_width=True, height=400)

        # Download
        out_csv = display_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Flagged Claims CSV",
            out_csv,
            f"fraud_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True,
        )

        # ── Top high-risk providers ──
        st.markdown('<div class="section-header">Top High-Risk Providers</div>', unsafe_allow_html=True)
        prov_risk = results_df.groupby("ProviderID").agg(
            Claims=("ClaimID","count"),
            AvgScore=("FraudScore(%)","mean"),
            FraudCount=("Verdict", lambda x: (x=="FRAUD").sum()),
        ).sort_values("AvgScore", ascending=False).head(10).reset_index()
        st.dataframe(prov_risk.style.format({"AvgScore":"{:.1f}%"})
                     .background_gradient(subset=["AvgScore"], cmap="RdYlGn_r"),
                     use_container_width=True)


# ═══════════════════════════════════════════════════
#  PAGE: Model Insights
# ═══════════════════════════════════════════════════
elif page == "📈 Model Insights":
    st.markdown("# 📈 Model Insights & Performance")
    st.divider()

    # Feature importance
    st.markdown('<div class="section-header">Feature Importance (Random Forest)</div>', unsafe_allow_html=True)
    feat_imp = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale=["#1d4ed8","#7c3aed","#ef4444"],
                     title="Feature Importances")
    fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", title_font_color="#e2e8f0",
                          coloraxis_showscale=False, height=420)
    st.plotly_chart(fig_imp, use_container_width=True)

    # Confusion matrix
    st.markdown('<div class="section-header">Model Performance on Test Set</div>', unsafe_allow_html=True)
    from sklearn.metrics import classification_report
    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        prec = report.get("1",{}).get("precision",0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Precision (Fraud)</div>
            <div class="metric-value">{prec*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        rec = report.get("1",{}).get("recall",0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">Recall (Fraud)</div>
            <div class="metric-value">{rec*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        f1 = report.get("1",{}).get("f1-score",0)
        st.markdown(f"""<div class="metric-card">
            <div class="metric-title">F1 Score (Fraud)</div>
            <div class="metric-value">{f1*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    # Dataset distributions
    st.markdown('<div class="section-header">Dataset Overview</div>', unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        fig_box = px.box(df, x="TreatmentType", y="BillingAmount",
                         color=df["FraudFlag"].map({0:"Legitimate",1:"Fraud"}),
                         color_discrete_map={"Fraud":"#ef4444","Legitimate":"#3b82f6"},
                         title="Billing Amount by Treatment Type")
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0", title_font_color="#e2e8f0")
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d2:
        fig_violin = px.violin(df, x="TreatmentType", y="TreatmentDurationDays",
                               color=df["FraudFlag"].map({0:"Legitimate",1:"Fraud"}),
                               color_discrete_map={"Fraud":"#ef4444","Legitimate":"#3b82f6"},
                               box=True, title="Treatment Duration Distribution")
        fig_violin.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                  font_color="#e2e8f0", title_font_color="#e2e8f0")
        st.plotly_chart(fig_violin, use_container_width=True)

    # Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Matrix</div>', unsafe_allow_html=True)
    num_cols = ["TreatmentDurationDays","BillingAmount","ApprovedAmount","NumProcedures",
                "OverchargeAmount","OverchargePct","BillingRatio","CostPerDay","CostPerProcedure","FraudFlag"]
    corr = df[num_cols].corr()
    fig_heat = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         title="Correlation Heatmap", text_auto=".2f")
    fig_heat.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                            title_font_color="#e2e8f0", height=500)
    st.plotly_chart(fig_heat, use_container_width=True)
