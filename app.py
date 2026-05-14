import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
tflite = tf.lite
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="NeuroScan AI - Radiology Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ---------------- CSS STYLING ----------------
st.markdown("""
<style>

body {
    background-color: #0b1220;
}

.main-header {
    font-size: 34px;
    font-weight: 800;
    color: #00d4ff;
}

.sub-header {
    font-size: 16px;
    color: #aab4c8;
}

.card {
    background-color: #111a2e;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,0,0,0.3);
}

.result {
    font-size: 28px;
    font-weight: 700;
    color: #00ffb3;
}

.warning {
    color: #ffcc00;
    font-weight: 600;
}

.danger {
    color: #ff4d4d;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="main-header">🏥 NeuroScan AI - Radiology Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered Brain Tumor Detection System for MRI Analysis</div>', unsafe_allow_html=True)

st.divider()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🧬 Clinical Info Panel")
    st.write("Supported MRI Classes:")
    st.markdown("""
    - Glioma Tumor  
    - Meningioma Tumor  
    - Pituitary Tumor  
    - No Tumor  
    """)

    st.info("Upload high-quality axial MRI scans for best accuracy.")

    st.warning("⚠️ This is an AI demo system, not a real medical device.")

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "brain_tumor_model.tflite"

interpreter = tf.lite.Interpreter(model_path="brain_tumor_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ---------------- UPLOAD SECTION ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 MRI Upload Panel")
    upload = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "png", "jpeg"])

# ---------------- PROCESSING ----------------
if upload is not None:

    img = Image.open(upload).convert("RGB")

    with col1:
        st.image(img, caption="🧠 Uploaded MRI Scan", use_container_width=True)

    st.markdown("### 🔬 AI Analysis Status")

    progress = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress.progress(i + 1)

    # preprocessing
    img_resized = img.resize((128, 128))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])[0]

    class_index = np.argmax(output)
    confidence = float(np.max(output)) * 100

    # ---------------- RESULT PANEL ----------------
    with col2:
        st.markdown("### 🧾 Radiology Report")

        if classes[class_index] == "No Tumor":
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<div class='result'>✅ Result: {classes[class_index]}</div>", unsafe_allow_html=True)
            st.markdown("Status: No abnormal tumor detected.")
            st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<div class='danger'>⚠️ Detected: {classes[class_index]}</div>", unsafe_allow_html=True)
            st.markdown("Status: Abnormal tissue detected in MRI scan.")
            st.markdown('</div>', unsafe_allow_html=True)

        st.write("### Confidence Level")
        st.progress(int(confidence))
        st.write(f"**{confidence:.2f}% certainty**")

# ---------------- FOOTER ----------------
st.divider()
st.caption("NeuroScan AI • Research Prototype • Not for clinical use")
