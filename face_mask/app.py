import streamlit as st
import numpy as np
from PIL import Image
import io
import time
import os

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MaskGuard AI",
    page_icon="😷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600;900&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }

.stApp {
    background: #070d1a;
    color: #e0e8ff;
}

.metric-card {
    background: linear-gradient(135deg, #0d1b2a 0%, #112240 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin: 8px 0;
}

.metric-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #00d4ff;
}

.metric-label {
    font-size: 0.75rem;
    color: #8899bb;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.result-box-mask {
    background: linear-gradient(135deg, #003d1f 0%, #00521f 100%);
    border: 2px solid #00c853;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-box-nomask {
    background: linear-gradient(135deg, #3d0000 0%, #520000 100%);
    border: 2px solid #ff1744;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
}

.result-title-mask   { font-family: 'Rajdhani', sans-serif; font-size: 2rem; color: #00e676; font-weight: 700; }
.result-title-nomask { font-family: 'Rajdhani', sans-serif; font-size: 2rem; color: #ff5252; font-weight: 700; }
.result-pct          { font-family: 'Rajdhani', sans-serif; font-size: 3rem; color: #00e676; font-weight: 900; }
.result-pct-no       { font-family: 'Rajdhani', sans-serif; font-size: 3rem; color: #ff5252; font-weight: 900; }

.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #00d4ff;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 6px;
    margin-bottom: 12px;
}

.info-banner {
    background: #0d1b2a;
    border-left: 4px solid #ffa726;
    padding: 12px 16px;
    border-radius: 0 8px 8px 0;
    color: #ffcc80;
    font-size: 0.85rem;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── model loading ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    """Load the trained Keras model. Cached so it only loads once."""
    # Try multiple paths so the app works locally and on Streamlit Cloud
    candidate_paths = [
        "best_facemask_model.keras",
        "face_mask/best_facemask_model.keras",
        os.path.join(os.path.dirname(__file__), "best_facemask_model.keras"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(p)
                return model, None
            except Exception as e:
                return None, str(e)
    return None, "Model file 'best_facemask_model.keras' not found. Please upload it to the repo."


def predict_mask(image: Image.Image, model) -> dict:
    """
    Run the trained CNN on a PIL image.
    Returns a dict with keys: label, confidence, with_mask_prob, no_mask_prob.
    """
    IMG_SIZE = 128

    # Preprocess exactly as during training
    img = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype=np.float32) / 255.0          # normalize to [0,1]
    arr = np.expand_dims(arr, axis=0)                       # (1, 128, 128, 3)

    raw_prob = float(model.predict(arr, verbose=0)[0][0])   # sigmoid output

    # The ImageDataGenerator sorts class folders alphabetically:
    # index 0 → with_mask, index 1 → without_mask  (w-i-t-h < w-i-t-h-o-u-t)
    # Adjust the interpretation based on what YOUR training printed for class_indices.
    # Default assumption: prob → P(without_mask)   i.e. class index 1
    # If your model printed {'with_mask': 0, 'without_mask': 1} keep this as-is.
    # If it printed {'with_mask': 1, ...} flip the logic below.
    no_mask_prob  = raw_prob
    with_mask_prob = 1.0 - raw_prob

    if with_mask_prob >= 0.5:
        label      = "with_mask"
        confidence = with_mask_prob
    else:
        label      = "without_mask"
        confidence = no_mask_prob

    return {
        "label":          label,
        "confidence":     confidence,
        "with_mask_prob": with_mask_prob,
        "no_mask_prob":   no_mask_prob,
    }


# ── sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 Model Intelligence")
    st.markdown("---")

    for val, lbl in [
        ("97.48%", "VAL ACCURACY"),
        ("0.9748",  "F1 SCORE"),
        ("0.9748",  "RECALL"),
        ("0.9977",  "ROC AUC"),
    ]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Per-Class Accuracy:**")
    st.markdown("🟢 With Mask: **97.5%**")
    st.markdown("🔴 Without Mask: **97.5%**")
    st.markdown("---")
    st.markdown("**Architecture:**")
    st.markdown("3× Conv blocks (32→64→128)  \nBatchNorm + MaxPool + Dropout  \nGlobalAvgPool → Dense → Sigmoid")
    st.markdown("---")
    st.caption("Task 5 — Binary CNN Classification")


# ── main title ──────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='font-family:Rajdhani,sans-serif; font-size:2.6rem; color:#00d4ff; margin-bottom:0;'>
😷 MaskGuard AI
</h1>
<p style='color:#8899bb; margin-top:4px;'>Face Mask Detection powered by a trained CNN (97.48% val accuracy)</p>
""", unsafe_allow_html=True)

# top metrics row
col1, col2, col3, col4, col5 = st.columns(5)
for col, val, lbl in zip(
    [col1, col2, col3, col4, col5],
    ["7,553", "97.48%", "0.9977", "128px", "Binary"],
    ["TOTAL IMAGES", "VAL ACCURACY", "ROC AUC", "INPUT SIZE", "TASK TYPE"],
):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{lbl}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── load model ──────────────────────────────────────────────────────────────────
model, model_error = load_model()

if model_error:
    st.markdown(f"""
    <div class="info-banner">
    ⚠️ <strong>Model not loaded:</strong> {model_error}<br>
    Make sure <code>best_facemask_model.keras</code> is committed to your GitHub repo
    alongside this <code>app.py</code>.
    </div>""", unsafe_allow_html=True)

# ── upload + predict ─────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="section-header">📸 Upload Face Image</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Choose an image (JPG / PNG / WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
        label_visibility="collapsed"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)

with right:
    st.markdown('<div class="section-header">🧠 CNN Analysis Result</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown("""
        <div style='background:#0d1b2a; border:1px dashed #1e3a5f; border-radius:16px;
                    padding:60px 20px; text-align:center; color:#4466aa;'>
            <div style='font-size:3rem;'>📤</div>
            <div style='margin-top:12px; font-size:1rem;'>Upload an image to analyse</div>
        </div>""", unsafe_allow_html=True)

    elif model is None:
        st.error("Cannot run inference — model failed to load. See the warning above.")

    else:
        with st.spinner("Running CNN inference…"):
            time.sleep(0.4)   # brief pause so the spinner is visible
            result = predict_mask(image, model)

        has_mask   = result["label"] == "with_mask"
        confidence = result["confidence"]
        box_cls    = "result-box-mask"   if has_mask else "result-box-nomask"
        title_cls  = "result-title-mask" if has_mask else "result-title-nomask"
        pct_cls    = "result-pct"        if has_mask else "result-pct-no"
        icon       = "✅" if has_mask else "❌"
        label_text = "MASK DETECTED"     if has_mask else "NO MASK DETECTED"
        sub_text   = "Face mask correctly worn · Compliant ✓" if has_mask else "No face mask detected · Non-Compliant ✗"

        st.markdown(f"""
        <div class="{box_cls}">
            <div style='font-size:3rem;'>{icon}</div>
            <div class="{title_cls}">{label_text}</div>
            <div class="{pct_cls}">{confidence*100:.1f}%</div>
            <div style='color:#aabbcc; font-size:0.85rem; margin-top:8px;'>MODEL CONFIDENCE</div>
            <div style='color:#aabbcc; font-size:0.8rem; margin-top:4px;'>{sub_text}</div>
        </div>""", unsafe_allow_html=True)

        # probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Probability breakdown**")
        st.progress(result["with_mask_prob"], text=f"With mask:    {result['with_mask_prob']*100:.1f}%")
        st.progress(result["no_mask_prob"],   text=f"Without mask: {result['no_mask_prob']*100:.1f}%")


# ── footer ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#445566; font-size:0.8rem;'>"
    "MaskGuard AI · CNN Face Mask Detection · Task 5 · Binary Classification"
    "</p>",
    unsafe_allow_html=True
)
