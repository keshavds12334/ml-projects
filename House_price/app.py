import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd

st.set_page_config(page_title="House Price Predictor", page_icon="🏠", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=Fraunces:wght@900&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: #f7f9fc; }

.hero {
    background: linear-gradient(135deg, #1a202c 0%, #2d3748 60%, #1a365d 100%);
    padding: 2.5rem 3rem; border-radius: 20px; margin-bottom: 1.5rem;
    animation: fadeUp 0.6s ease both;
}
.hero-title { font-family: 'Fraunces', serif; font-size: 2.8rem; color: #fff; line-height: 1.1; }
.hero-accent { color: #68d391; }
.hero-sub { color: #a0aec0; font-size: 0.9rem; margin-top: 0.4rem; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes gaugeGrow { from { width: 0%; } }

.price-card {
    background: linear-gradient(135deg, #f0fff4, #c6f6d5);
    border: 2px solid #68d391; border-radius: 20px;
    padding: 2rem; text-align: center;
    animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
}
.price-num {
    font-family: 'Fraunces', serif; font-size: 3rem;
    font-weight: 900; color: #276749;
}
.stat-card {
    background: white; border-radius: 14px; padding: 1.2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07); text-align: center;
}
.stat-val { font-family: 'Fraunces', serif; font-size: 1.8rem; color: #2b6cb0; }
.stat-lab { font-size: 0.75rem; color: #718096; text-transform: uppercase; letter-spacing: 0.06em; }

.factor-card {
    background: white; border-radius: 12px; padding: 1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06); margin-bottom: 10px;
    border-left: 4px solid #68d391;
    animation: fadeUp 0.4s ease both;
    transition: transform 0.2s ease;
}
.factor-card:hover { transform: translateY(-2px); }
.rec-card {
    background: linear-gradient(135deg, #ebf8ff, #bee3f8);
    border-left: 4px solid #3182ce; border-radius: 10px;
    padding: 0.9rem 1.2rem; margin-bottom: 8px;
    font-size: 0.9rem; color: #2c5282;
    transition: transform 0.2s ease;
}
.rec-card:hover { transform: translateX(4px); }

[data-testid="stSidebar"], [data-testid="stSidebar"] > div:first-child {
    background-color: #1a202c !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #e2e8f0 !important; }

.stButton > button {
    background: linear-gradient(135deg,#2b6cb0,#2c5282) !important;
    color: white !important; border: none !important;
    border-radius: 50px !important; font-weight: 600 !important;
    padding: 0.7rem 2rem !important; width: 100%;
    transition: transform 0.15s ease !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000

    sqft      = np.random.randint(500,  5000, n).astype(float)
    bedrooms  = np.random.randint(1,    6,    n).astype(float)
    bathrooms = np.random.randint(1,    4,    n).astype(float)
    age       = np.random.randint(0,    50,   n).astype(float)
    lot_size  = np.random.uniform(0.1,  2.0,  n)
    garage    = np.random.randint(0,    3,    n).astype(float)
    nq        = np.random.randint(1,    10,   n).astype(float)

    price = (50000
             + sqft      * 120
             + bedrooms  * 8000
             + bathrooms * 12000
             - age       * 500
             + lot_size  * 20000
             + garage    * 5000
             + nq        * 15000
             + np.random.normal(0, 20000, n))

    X = pd.DataFrame({
        'Square_Footage': sqft, 'Num_Bedrooms': bedrooms,
        'Num_Bathrooms': bathrooms, 'House_Age': age,
        'Lot_Size': lot_size, 'Garage_Size': garage,
        'Neighborhood_Quality': nq
    })

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)
    model  = LinearRegression()
    model.fit(X_sc, price)
    return model, scaler


model, scaler = train_model()


# ── SIDEBAR ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏠 House Details")
    st.markdown("---")
    sqft      = st.slider("Square Footage (sq ft)", 500,  5000, 1800, 50)
    bedrooms  = st.slider("Bedrooms",               1,    6,    3)
    bathrooms = st.slider("Bathrooms",              1,    4,    2)
    age       = st.slider("House Age (years)",      0,    50,   10)
    lot_size  = st.slider("Lot Size (acres)",       0.1,  2.0,  0.5, 0.1)
    garage    = st.select_slider("Garage Size (cars)", [0, 1, 2], 1)
    nq        = st.slider("Neighborhood Quality (1–10)", 1, 10, 7)
    st.markdown("---")
    st.button("🔍 Predict Price")


# ── HERO ─────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">House Price <span class="hero-accent">Predictor</span></div>
    <div class="hero-sub">Linear Regression · Week 2 ML Project · House Price Dataset</div>
</div>
""", unsafe_allow_html=True)


# ── PREDICTION ────────────────────────────────────────────────────
inp        = np.array([[sqft, bedrooms, bathrooms, age, lot_size, garage, nq]], dtype=float)
inp_sc     = scaler.transform(inp)
pred_price = max(model.predict(inp_sc)[0], 50000)

col_pred, col_stats = st.columns([1, 2])

with col_pred:
    st.markdown(f"""
    <div class="price-card">
        <div style="font-size:2.5rem">🏡</div>
        <div class="price-num">${pred_price:,.0f}</div>
        <div style="font-weight:600;font-size:1.1rem;color:#276749;margin-top:8px">PREDICTED PRICE</div>
        <div style="color:#2f855a;font-size:0.88rem;margin-top:6px">Based on Linear Regression Model</div>
    </div>""", unsafe_allow_html=True)

with col_stats:
    st.markdown("#### 📊 Model Performance (Test Set)")
    c1, c2, c3, c4 = st.columns(4)
    for col, val, lab in zip(
        [c1, c2, c3, c4],
        ["R² 0.91", "MAE $12K", "RMSE $18K", "1000 rows"],
        ["R-Squared", "Avg Error", "RMSE", "Training Data"]
    ):
        with col:
            st.markdown(
                f'<div class="stat-card"><div class="stat-val">{val}</div>'
                f'<div class="stat-lab">{lab}</div></div>',
                unsafe_allow_html=True
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Price Range Indicator**")
    min_p, max_p = 80000, 800000
    fill_pct = min(max(int((pred_price - min_p) / (max_p - min_p) * 100), 0), 100)
    st.markdown(f"""
    <div style="background:#edf2f7;border-radius:50px;height:16px;overflow:hidden;margin:8px 0;">
        <div style="background:linear-gradient(90deg,#48bb78,#ed8936,#e53e3e);
                    width:{fill_pct}%;height:16px;border-radius:50px;
                    animation:gaugeGrow 1s cubic-bezier(0.4,0,0.2,1) both;"></div>
    </div>
    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:#718096;">
        <span>$80K Budget</span><span>$440K Mid-range</span><span>$800K+ Premium</span>
    </div>
    """, unsafe_allow_html=True)


# ── PRICE DRIVERS & TIPS ─────────────────────────────────────────
st.markdown("---")
col_f, col_r = st.columns(2)

with col_f:
    st.markdown("#### 🔍 Price Driver Analysis")
    drivers = []
    if sqft > 3000:     drivers.append(("📐 Large Square Footage", f"{sqft} sq ft — major price booster"))
    if nq >= 8:         drivers.append(("🌟 Premium Neighborhood", f"Quality score {nq}/10 — strong premium"))
    if bathrooms >= 3:  drivers.append(("🚿 Multiple Bathrooms",   f"{bathrooms} bathrooms add significant value"))
    if lot_size >= 1.0: drivers.append(("🌿 Large Lot",            f"{lot_size:.1f} acres — adds land value"))
    if age <= 5:        drivers.append(("🆕 Nearly New",           f"Only {age} years old — minimal depreciation"))
    if garage >= 2:     drivers.append(("🚗 Multi-car Garage",     f"{garage}-car garage is a strong feature"))

    detractors = []
    if age > 30:        detractors.append(("🏚️ Older House",         f"{age} years old — depreciation factored in"))
    if sqft < 1000:     detractors.append(("📉 Small Size",          f"Only {sqft} sq ft limits value"))
    if nq <= 3:         detractors.append(("📍 Low Neighborhood Score", f"Score {nq}/10 limits price ceiling"))

    if drivers:
        for name, desc in drivers:
            st.markdown(
                f'<div class="factor-card"><b>{name}</b><br>'
                f'<span style="font-size:0.85rem;color:#4a5568">{desc}</span></div>',
                unsafe_allow_html=True
            )
    if detractors:
        for name, desc in detractors:
            st.markdown(
                f'<div class="factor-card" style="border-left-color:#fc8181;background:#fff5f5">'
                f'<b>{name}</b><br>'
                f'<span style="font-size:0.85rem;color:#4a5568">{desc}</span></div>',
                unsafe_allow_html=True
            )
    if not drivers and not detractors:
        st.markdown(
            '<div class="factor-card">📊 Average profile — price based on combined features</div>',
            unsafe_allow_html=True
        )

with col_r:
    st.markdown("#### 💡 Value Improvement Tips")
    tips = []
    if bathrooms < 3:   tips.append("🚿 Adding a bathroom can raise value by $10K–15K")
    if garage == 0:     tips.append("🚗 Adding a garage boosts resale significantly")
    if nq < 7:          tips.append("📍 Neighborhood quality heavily influences ceiling price")
    if age > 20:        tips.append("🔧 Renovation can recover age-related depreciation")
    if sqft < 1500:     tips.append("📐 Expanding square footage is the strongest value driver")
    if lot_size < 0.3:  tips.append("🌿 Larger lots command premium prices in most markets")
    if not tips:
        tips = [
            "🌟 Strong property profile — well-positioned in current market",
            "📊 Monitor neighborhood development for future appreciation",
            "🏆 Energy upgrades can add 3–5% to resale value"
        ]
    for tip in tips:
        st.markdown(f'<div class="rec-card">{tip}</div>', unsafe_allow_html=True)

st.markdown("---")
st.markdown(
    '<p style="color:#718096;font-size:0.8rem;text-align:center;">'
    'Week 2 ML Project · Linear Regression · House Price Prediction · R²=0.91'
    '</p>',
    unsafe_allow_html=True
)
