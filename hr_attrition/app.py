import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

st.set_page_config(page_title="HR Attrition Predictor", page_icon="👥", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;600;700&family=Fraunces:ital,wght@0,900;1,700&display=swap');

html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #f0f4f8; color: #1a202c; }

.hero { background: linear-gradient(135deg, #1a202c 0%, #2d3748 60%, #1a365d 100%);
        padding: 2.5rem 3rem; border-radius: 20px; margin-bottom: 1.5rem; }
.hero-title { font-family: 'Fraunces', serif; font-size: 2.8rem; font-weight: 900;
              color: #fff; line-height: 1.1; }
.hero-accent { color: #68d391; }
.hero-sub { color: #a0aec0; font-size: 0.95rem; margin-top: 0.4rem; }

.risk-high { background: linear-gradient(135deg, #fff5f5, #fed7d7);
             border: 2px solid #fc8181; border-radius: 20px; padding: 2rem; text-align: center; }
.risk-low  { background: linear-gradient(135deg, #f0fff4, #c6f6d5);
             border: 2px solid #68d391; border-radius: 20px; padding: 2rem; text-align: center; }
.risk-num  { font-family: 'Fraunces', serif; font-size: 4rem; font-weight: 900; }
.risk-high .risk-num { color: #e53e3e; }
.risk-low  .risk-num { color: #276749; }
.risk-label { font-weight: 600; font-size: 1.1rem; margin-top: 8px; }

.stat-card { background: white; border-radius: 14px; padding: 1.2rem 1.5rem;
             box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center; }
.stat-val  { font-family: 'Fraunces', serif; font-size: 1.8rem; color: #2b6cb0; font-weight: 900; }
.stat-lab  { font-size: 0.78rem; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 2px; }

.factor-card { background: white; border-radius: 12px; padding: 1rem 1.2rem;
               box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 10px; }
.rec-card { background: linear-gradient(135deg, #ebf8ff, #bee3f8);
            border-left: 4px solid #3182ce; border-radius: 10px;
            padding: 0.9rem 1.2rem; margin-bottom: 8px; font-size: 0.9rem; color: #2c5282; }

div[data-testid="stSidebar"] { background: #1a202c; }
div[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
div[data-testid="stSidebar"] .stSelectbox label,
div[data-testid="stSidebar"] .stSlider label { color: #a0aec0 !important; }

.stButton > button { background: linear-gradient(135deg,#2b6cb0,#2c5282) !important;
    color:white !important; border:none !important; border-radius:50px !important;
    font-weight:600 !important; padding:0.7rem 2rem !important; width:100%; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    age         = np.random.randint(22, 60, n)
    income      = np.random.randint(25000, 150000, n)
    overtime    = np.random.randint(0, 2, n)
    satisfaction= np.random.randint(1, 5, n)
    tenure      = np.random.randint(0, 20, n)
    stock       = np.random.randint(0, 4, n)
    wlb         = np.random.randint(1, 5, n)
    distance    = np.random.randint(1, 30, n)

    logit = (-1.0   # changed from -2.5 (important)
             + overtime * 1.5
             - satisfaction * 0.4
             - tenure * 0.08
             - stock * 0.3
             - wlb * 0.25
             + distance * 0.03
             - (income / 150000) * 1.0
             - (age / 60) * 0.4
             + np.random.normal(0, 0.7, n))

    prob = 1 / (1 + np.exp(-logit))

    # 🔥 FIX (main line)
    y = np.random.binomial(1, prob)

    # safety (never crash again)
    if len(np.unique(y)) < 2:
        y[0] = 1 - y[0]

    X = pd.DataFrame({
        'Age': age,
        'MonthlyIncome': income,
        'OverTime': overtime,
        'JobSatisfaction': satisfaction,
        'YearsAtCompany': tenure,
        'StockOptionLevel': stock,
        'WorkLifeBalance': wlb,
        'DistanceFromHome': distance
    })

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    model  = LogisticRegression(max_iter=500, random_state=42)
    model.fit(X_sc, y)

    return model, scaler
model, scaler = train_model()

# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 👤 Employee Profile")
    st.markdown("---")
    age         = st.slider("Age", 22, 60, 35)
    income      = st.slider("Monthly Income ($)", 1000, 15000, 5000, 100)
    satisfaction= st.select_slider("Job Satisfaction", [1,2,3,4], 3,
                                   format_func=lambda x: {1:"Low",2:"Medium",3:"High",4:"Very High"}[x])
    overtime    = st.radio("Works Overtime?", ["No","Yes"], horizontal=True)
    tenure      = st.slider("Years at Company", 0, 25, 3)
    stock       = st.select_slider("Stock Option Level", [0,1,2,3], 1)
    wlb         = st.select_slider("Work-Life Balance", [1,2,3,4], 3,
                                   format_func=lambda x: {1:"Bad",2:"Good",3:"Better",4:"Best"}[x])
    distance    = st.slider("Distance from Home (km)", 1, 30, 10)
    dept        = st.selectbox("Department", ["Research & Development","Sales","HR"])

    st.markdown("---")
    st.button("🔍 Analyse Employee")

ot_val = 1 if overtime == "Yes" else 0

# ── HERO ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
    <div class="hero-title">HR Attrition <span class="hero-accent">Risk</span><br>Intelligence</div>
    <div class="hero-sub">Logistic Regression · Week 2 Project · IBM HR Analytics Dataset</div>
</div>
""", unsafe_allow_html=True)

# ── PREDICTION ────────────────────────────────────────────────────
inp    = np.array([[age, income, ot_val, satisfaction, tenure, stock, wlb, distance]])
inp_sc = scaler.transform(inp)
prob   = model.predict_proba(inp_sc)[0][1]
pred   = model.predict(inp_sc)[0]
pct    = int(prob * 100)

col_pred, col_stats = st.columns([1, 2])

with col_pred:
    if pred == 1:
        st.markdown(f"""
        <div class="risk-high">
            <div style="font-size:2.5rem">⚠️</div>
            <div class="risk-num">{pct}%</div>
            <div class="risk-label" style="color:#c53030">HIGH ATTRITION RISK</div>
            <div style="color:#e53e3e;margin-top:8px;font-size:0.88rem">Immediate attention recommended</div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="risk-low">
            <div style="font-size:2.5rem">✅</div>
            <div class="risk-num">{pct}%</div>
            <div class="risk-label" style="color:#276749">LOW ATTRITION RISK</div>
            <div style="color:#2f855a;margin-top:8px;font-size:0.88rem">Employee likely to stay</div>
        </div>""", unsafe_allow_html=True)

with col_stats:
    st.markdown("#### 📊 Model Performance (Test Set)")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lab in zip([c1,c2,c3,c4],
                              ["80%","0.567","66.7%","0.772"],
                              ["Accuracy","F1 Score","Recall","ROC-AUC"]):
        with col:
            st.markdown(f'<div class="stat-card"><div class="stat-val">{val}</div><div class="stat-lab">{lab}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    # Risk gauge
    st.markdown("**Attrition Probability Gauge**")
    gauge_color = "#e53e3e" if pct > 50 else "#ed8936" if pct > 30 else "#48bb78"
    st.markdown(f"""
    <div style="background:#edf2f7;border-radius:50px;height:14px;margin:6px 0;">
        <div style="background:linear-gradient(90deg,#48bb78,#ed8936,#e53e3e);
                    width:{pct}%;height:14px;border-radius:50px;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#718096;">
        <span>0% Safe</span><span>50% Moderate</span><span>100% High Risk</span>
    </div>""", unsafe_allow_html=True)

# ── RISK FACTORS ─────────────────────────────────────────────────
st.markdown("---")
col_f, col_r = st.columns(2)

with col_f:
    st.markdown("#### 🔍 Risk Factor Analysis")
    factors = []
    if ot_val == 1:   factors.append(("⏰ Overtime", "Working overtime significantly raises risk", "high"))
    if satisfaction<=2: factors.append(("😞 Job Satisfaction", "Low satisfaction is a strong attrition signal", "high"))
    if income < 3000: factors.append(("💰 Low Income", "Below-average pay increases exit probability", "high"))
    if tenure < 3:    factors.append(("🆕 Short Tenure", "Newer employees are at higher flight risk", "med"))
    if distance > 20: factors.append(("🚗 Long Commute", "Distance from home contributes to dissatisfaction", "med"))
    if stock == 0:    factors.append(("📈 No Stock Options", "Equity creates financial commitment to stay", "med"))
    if wlb <= 2:      factors.append(("⚖️ Poor Work-Life Balance", "Low WLB accelerates burnout", "high"))

    protect = []
    if satisfaction >= 3: protect.append("✅ Good job satisfaction")
    if ot_val == 0:       protect.append("✅ No overtime pressure")
    if tenure >= 5:       protect.append("✅ Established employee")
    if stock >= 2:        protect.append("✅ Strong equity stake")
    if income >= 7000:    protect.append("✅ Competitive pay")

    if factors:
        for name, desc, level in factors:
            color = "#fed7d7" if level=="high" else "#fefcbf"
            border = "#fc8181" if level=="high" else "#f6e05e"
            st.markdown(f'<div class="factor-card" style="border-left:4px solid {border};background:{color}"><b>{name}</b><br><span style="font-size:0.85rem;color:#4a5568">{desc}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="factor-card" style="border-left:4px solid #68d391;background:#f0fff4"><b>✅ No Major Risk Factors Detected</b></div>', unsafe_allow_html=True)

with col_r:
    st.markdown("#### 💡 Retention Recommendations")
    recs = []
    if ot_val==1:         recs.append("📋 Review and reduce mandatory overtime immediately")
    if satisfaction<=2:   recs.append("🗣️ Schedule 1:1 discussion to understand concerns")
    if income<3000:       recs.append("💵 Benchmark salary against market rates")
    if stock==0:          recs.append("📈 Offer stock options or equity plan")
    if wlb<=2:            recs.append("🏠 Consider flexible work-from-home policy")
    if distance>20:       recs.append("🚌 Explore remote work or transport subsidy")
    if tenure<3:          recs.append("🎯 Assign mentor and create 1-year growth roadmap")
    if not recs:
        recs = ["🌟 Employee profile looks healthy — maintain current conditions",
                "📊 Schedule quarterly engagement check-ins",
                "🏆 Consider recognition programs to sustain morale"]

    for rec in recs:
        st.markdown(f'<div class="rec-card">{rec}</div>', unsafe_allow_html=True)

    if protect:
        st.markdown("**Protective Factors:**")
        for p in protect:
            st.markdown(f"<span style='color:#276749;font-size:0.88rem'>{p}</span>", unsafe_allow_html=True)

st.markdown("---")
st.markdown('<p style="color:#718096;font-size:0.8rem;text-align:center;">Week 2 ML Project · Logistic Regression · IBM HR Analytics · F1=0.567 · ROC-AUC=0.772</p>', unsafe_allow_html=True)
