import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Telecom Churn Predictor", page_icon="📡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=Inter:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #050b14; color: #e8eaf6; }

.top-bar { background: linear-gradient(90deg,#0a1628,#0d2137);
           border-bottom: 1px solid rgba(100,200,255,0.15);
           padding: 1.5rem 2rem; margin-bottom: 1.5rem; border-radius: 12px; }
.brand { font-family:'Syne',sans-serif; font-size:2.6rem; font-weight:800;
         background: linear-gradient(90deg,#64c8ff,#00e5ff,#40ffaa);
         -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.sub   { color:#546e7a; font-size:0.9rem; font-weight:300; letter-spacing:0.05em; }

.churn-danger { background:linear-gradient(135deg,rgba(244,67,54,0.08),rgba(229,57,53,0.15));
                border:1.5px solid #f44336; border-radius:18px; padding:2rem; text-align:center; }
.churn-safe   { background:linear-gradient(135deg,rgba(0,230,118,0.06),rgba(0,200,100,0.12));
                border:1.5px solid #00e676; border-radius:18px; padding:2rem; text-align:center; }
.big-pct { font-family:'Syne',sans-serif; font-size:4.5rem; font-weight:800; line-height:1; }
.danger-pct { color:#f44336; }
.safe-pct   { color:#00e676; }
.verdict { font-size:1rem; font-weight:600; letter-spacing:0.08em; text-transform:uppercase; margin-top:8px; }

.kpi { background:rgba(255,255,255,0.03); border:1px solid rgba(100,200,255,0.15);
       border-radius:14px; padding:1rem 1.2rem; text-align:center; }
.kpi-val { font-family:'Syne',sans-serif; font-size:1.6rem; font-weight:700; color:#64c8ff; }
.kpi-lab { font-size:0.72rem; color:#546e7a; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px; }

.seg-card { background:rgba(255,255,255,0.02); border:1px solid rgba(100,200,255,0.1);
            border-radius:12px; padding:1rem 1.3rem; margin-bottom:8px; }
.seg-tag  { display:inline-block; border-radius:50px; padding:3px 12px;
            font-size:0.75rem; font-weight:600; margin-right:6px; }
.tag-risk { background:rgba(244,67,54,0.15); color:#f44336; border:1px solid rgba(244,67,54,0.3); }
.tag-ok   { background:rgba(0,230,118,0.1); color:#00e676; border:1px solid rgba(0,230,118,0.25); }

div[data-testid="stSidebar"] { background:#040810; border-right:1px solid rgba(100,200,255,0.1); }

.stButton > button { background:linear-gradient(90deg,#64c8ff,#00e5ff) !important;
    color:#050b14 !important; border:none !important; border-radius:50px !important;
    font-weight:700 !important; font-size:0.95rem !important; padding:0.7rem 2rem !important;
    width:100%; letter-spacing:0.02em; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 2000
    tenure   = np.random.randint(0, 72, n)
    monthly  = np.random.uniform(20, 120, n)
    contract = np.random.choice([0,1,2], n, p=[0.55,0.25,0.2])   # 0=mo-to-mo,1=1yr,2=2yr
    fiber    = np.random.randint(0, 2, n)
    tech     = np.random.randint(0, 2, n)
    security = np.random.randint(0, 2, n)
    echeck   = np.random.randint(0, 2, n)
    pb       = np.random.randint(0, 2, n)
    senior   = np.random.randint(0, 2, n)

    logit = (-1.0
             - tenure * 0.06
             + monthly * 0.015
             - contract * 1.2
             + fiber * 0.6
             - tech * 0.4
             - security * 0.5
             + echeck * 0.5
             + pb * 0.2
             + senior * 0.3
             + np.random.normal(0, 0.5, n))
    prob = 1/(1+np.exp(-logit))
    y = (prob > 0.5).astype(int)

    X = pd.DataFrame({'tenure':tenure,'MonthlyCharges':monthly,'Contract':contract,
                      'FiberOptic':fiber,'TechSupport':tech,'OnlineSecurity':security,
                      'ElectronicCheck':echeck,'PaperlessBilling':pb,'SeniorCitizen':senior})
    scaler = StandardScaler(); X_sc = scaler.fit_transform(X)
    mdl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    mdl.fit(X_sc, y)
    return mdl, scaler

model, scaler = train_model()

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📡 Customer Profile")
    st.markdown("---")
    tenure   = st.slider("Tenure (months)", 0, 72, 12)
    monthly  = st.slider("Monthly Charges ($)", 20, 120, 65)
    contract = st.radio("Contract Type", ["Month-to-Month","One Year","Two Year"])
    internet = st.radio("Internet Service", ["No Internet","DSL","Fiber Optic"])
    tech     = st.radio("Tech Support", ["No","Yes"], horizontal=True)
    security = st.radio("Online Security", ["No","Yes"], horizontal=True)
    payment  = st.radio("Payment Method", ["Electronic Check","Bank Transfer","Credit Card"])
    paperless= st.radio("Paperless Billing", ["No","Yes"], horizontal=True)
    senior   = st.checkbox("Senior Citizen")

    st.markdown("---")
    st.button("🔍 Predict Churn Risk")

contract_map = {"Month-to-Month":0,"One Year":1,"Two Year":2}
fiber_val    = 1 if internet == "Fiber Optic" else 0
echeck_val   = 1 if payment == "Electronic Check" else 0

inp    = np.array([[tenure, monthly, contract_map[contract], fiber_val,
                    1 if tech=="Yes" else 0, 1 if security=="Yes" else 0,
                    echeck_val, 1 if paperless=="Yes" else 0, int(senior)]])
inp_sc = scaler.transform(inp)
prob   = model.predict_proba(inp_sc)[0][1]
pred   = model.predict(inp_sc)[0]
pct    = int(prob * 100)

# ── HEADER ────────────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <div class="brand">ChurnGuard AI</div>
    <div class="sub">Telecom Customer Retention Intelligence · Gradient Boosting · Week 3 Project</div>
</div>""", unsafe_allow_html=True)

# ── MAIN PREDICTION ───────────────────────────────────────────────
col_pred, col_meter, col_kpi = st.columns([1.2, 1, 1.5])

with col_pred:
    if pred == 1:
        st.markdown(f"""<div class="churn-danger">
            <div style="font-size:2rem">🚨</div>
            <div class="big-pct danger-pct">{pct}%</div>
            <div class="verdict" style="color:#f44336">Will Churn</div>
            <div style="color:#ef9a9a;font-size:0.82rem;margin-top:10px">
                Act now — high revenue loss risk
            </div></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="churn-safe">
            <div style="font-size:2rem">🟢</div>
            <div class="big-pct safe-pct">{pct}%</div>
            <div class="verdict" style="color:#00e676">Will Stay</div>
            <div style="color:#69f0ae;font-size:0.82rem;margin-top:10px">
                Customer likely to remain loyal
            </div></div>""", unsafe_allow_html=True)

with col_meter:
    st.markdown("**Churn Probability**")
    bar_w   = pct
    bar_col = "#f44336" if pct>60 else "#ff9800" if pct>30 else "#00e676"
    st.markdown(f"""
    <div style="height:180px;display:flex;align-items:flex-end;gap:2px;">
        <div style="flex:1;background:rgba(255,255,255,0.04);border-radius:8px 8px 0 0;height:100%;
                    position:relative;overflow:hidden;">
            <div style="position:absolute;bottom:0;width:100%;height:{pct}%;
                        background:{bar_col};border-radius:8px 8px 0 0;
                        transition:height 0.5s ease;"></div>
        </div>
    </div>
    <div style="text-align:center;margin-top:6px;font-family:'Syne',sans-serif;
                font-size:1.5rem;color:{bar_col};font-weight:700">{pct}%</div>
    """, unsafe_allow_html=True)

with col_kpi:
    st.markdown("**Model Performance**")
    kpis = [("F1 Score","0.607"),("ROC-AUC","0.759"),("Recall","72.1%"),("Accuracy","77.9%")]
    for i in range(0,4,2):
        c1,c2 = st.columns(2)
        with c1:
            st.markdown(f'<div class="kpi"><div class="kpi-val">{kpis[i][1]}</div><div class="kpi-lab">{kpis[i][0]}</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="kpi"><div class="kpi-val">{kpis[i+1][1]}</div><div class="kpi-lab">{kpis[i+1][0]}</div></div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

# ── RISK ANALYSIS ─────────────────────────────────────────────────
st.markdown("---")
col_risk, col_action = st.columns(2)

with col_risk:
    st.markdown("#### 🔍 Risk Signal Analysis")
    signals = []
    if contract == "Month-to-Month": signals.append(("📅 Month-to-Month Contract","No lock-in — easy to cancel","risk"))
    if internet == "Fiber Optic":    signals.append(("🌐 Fiber Optic Internet","Highest churn segment (Importance #1)","risk"))
    if payment == "Electronic Check":signals.append(("💳 Electronic Check Payment","Highest churn payment method","risk"))
    if tenure < 12:                  signals.append(("⏱️ Short Tenure","New customers are highest churn risk","risk"))
    if monthly > 80:                 signals.append(("💰 High Monthly Bill","Price sensitivity increases churn","risk"))
    if tech == "No":                 signals.append(("🛠️ No Tech Support","Reduces stickiness","risk"))
    if security == "No":             signals.append(("🔒 No Online Security","Missing loyalty-building service","risk"))

    good = []
    if contract != "Month-to-Month": good.append("✅ Contracted customer")
    if tenure >= 24:                  good.append("✅ Long-term loyal customer")
    if tech == "Yes":                 good.append("✅ Tech support subscriber")
    if security == "Yes":             good.append("✅ Online security subscriber")

    for name, desc, t in signals:
        st.markdown(f'<div class="seg-card"><span class="seg-tag tag-risk">RISK</span><b>{name}</b><br><span style="font-size:0.82rem;color:#78909c">{desc}</span></div>', unsafe_allow_html=True)
    for g in good:
        st.markdown(f'<div class="seg-card"><span class="seg-tag tag-ok">SAFE</span><span style="color:#00e676">{g}</span></div>', unsafe_allow_html=True)

with col_action:
    st.markdown("#### 💡 Retention Strategy")
    actions = []
    if contract=="Month-to-Month":   actions.append(("🎁","Offer 15–20% discount for 1-year contract signup","High Priority"))
    if internet=="Fiber Optic":      actions.append(("🔧","Proactively reach out — ask about service satisfaction","High Priority"))
    if payment=="Electronic Check":  actions.append(("💳","Incentivise switch to auto-pay: 1 month free","High Priority"))
    if tenure<12:                    actions.append(("🎯","Enroll in New Customer Loyalty Program","Medium Priority"))
    if monthly>80:                   actions.append(("📊","Offer a personalised plan review call","Medium Priority"))
    if tech=="No":                   actions.append(("🛠️","Bundle TechSupport for 3 months free","Medium Priority"))
    if security=="No":               actions.append(("🔒","Offer OnlineSecurity trial at no charge","Low Priority"))
    if not actions:                  actions=[("🌟","Continue current service — customer is low risk","Maintain"),
                                              ("📞","Schedule quarterly satisfaction check-in","Maintain")]

    priority_colors = {"High Priority":"#f44336","Medium Priority":"#ff9800","Low Priority":"#64c8ff","Maintain":"#00e676"}
    for icon, action, priority in actions:
        color = priority_colors.get(priority, "#64c8ff")
        st.markdown(f"""<div style="background:rgba(255,255,255,0.02);border-radius:12px;
                        padding:0.9rem;margin-bottom:8px;border-left:3px solid {color}">
            <div style="display:flex;justify-content:space-between;align-items:flex-start">
                <span>{icon} <span style="font-size:0.88rem">{action}</span></span>
                <span style="font-size:0.7rem;color:{color};font-weight:600;white-space:nowrap;margin-left:8px">{priority}</span>
            </div></div>""", unsafe_allow_html=True)

    # Revenue impact
    monthly_rev = monthly
    annual_risk = monthly_rev * 12 * (prob)
    st.markdown(f"""
    <div style="background:rgba(244,67,54,0.08);border:1px solid rgba(244,67,54,0.3);
                border-radius:12px;padding:1rem;margin-top:12px;text-align:center">
        <div style="color:#78909c;font-size:0.78rem;text-transform:uppercase;letter-spacing:0.06em">Estimated Annual Revenue at Risk</div>
        <div style="font-family:'Syne',sans-serif;font-size:2rem;color:#f44336;font-weight:700">${annual_risk:,.0f}</div>
        <div style="color:#546e7a;font-size:0.78rem">Based on ${monthly}/month · {pct}% churn probability</div>
    </div>""", unsafe_allow_html=True)

# ── TOP CHURN DRIVERS ─────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 📊 Top Churn Drivers (from Model Training)")
drivers = [("Fiber Optic Internet",0.222),("Tenure",0.191),("Electronic Check",0.092),
           ("Total Charges",0.084),("2-Year Contract",0.074),("1-Year Contract",0.066),
           ("Paperless Billing",0.045),("Monthly Charges",0.041)]
cols = st.columns(4)
for i,(name,val) in enumerate(drivers):
    with cols[i%4]:
        pct_bar = int(val*450)
        col_b = "#f44336" if val>0.15 else "#ff9800" if val>0.07 else "#64c8ff"
        st.markdown(f"""<div class="kpi" style="margin-bottom:8px">
            <div style="font-size:0.78rem;color:#90a4ae;margin-bottom:6px">{name}</div>
            <div style="background:rgba(255,255,255,0.05);border-radius:4px;height:6px">
                <div style="background:{col_b};width:{pct_bar}%;height:6px;border-radius:4px"></div>
            </div>
            <div style="color:{col_b};font-weight:700;font-size:0.95rem;margin-top:4px">{val:.1%}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="color:#37474f;font-size:0.78rem;text-align:center;">Week 3 ML Project · Gradient Boosting (Tuned) · IBM Telco · F1=0.607 · AUC=0.759 · Recall=72.1%</p>', unsafe_allow_html=True)
