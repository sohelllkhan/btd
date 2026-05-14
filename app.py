import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
tflite = tf.lite
import time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO
#--- tumor spot ---
def create_heatmap(img):
    img = np.array(img.resize((128,128)))

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return overlay
# ---- pdf -----
def generate_pdf(name, age, gender, pid, result, confidence):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, 800, "NeuroScan AI - Medical Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, 760, f"Patient Name: {name}")
    c.drawString(50, 740, f"Age: {age}")
    c.drawString(50, 720, f"Gender: {gender}")
    c.drawString(50, 700, f"Patient ID: {pid}")

    c.drawString(50, 660, f"Diagnosis: {result}")
    c.drawString(50, 640, f"Confidence: {confidence:.2f}%")

    c.drawString(50, 600, "Note: This is an AI-generated report, not a final medical diagnosis.")

    c.save()
    buffer.seek(0)
    return buffer

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

# ------- patient info ------
# ---------------- PATIENT REGISTRATION ----------------
st.markdown("## 🧑‍⚕️ Patient Information System")

colA, colB, colC, colD = st.columns(4)

with colA:
    patient_name = st.text_input("Patient Name")

with colB:
    age = st.number_input("Age", min_value=0, max_value=120)

with colC:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])

with colD:
    patient_id = st.text_input("Patient ID")

st.divider()

# ---------------- UPLOAD SECTION ----------------
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 MRI Upload Panel")
    upload = st.file_uploader("Upload Brain MRI Scan", type=["jpg", "png", "jpeg"])

# ---------------- PROCESSING ----------------
if upload is not None:

    st.markdown("### 📋 Patient Medical Record")
    st.write(f"**Name:** {patient_name if patient_name else 'Unknown'}")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Patient ID:** {patient_id if patient_id else 'N/A'}")
    st.divider()

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
    # --- img ---
    st.markdown("### 🔥 AI Attention Heatmap")
    heatmap_img = create_heatmap(img)
    st.image(
    heatmap_img,
    caption="AI Focus Regions (Simulated Heatmap)",
    use_container_width=True)

    # prediction
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    pdf = generate_pdf(
    patient_name if patient_name else "Unknown",
    age,
    gender,
    patient_id if patient_id else "N/A",
    classes[class_index],
    confidence)
    st.download_button(
    label="📄 Download Medical Report (PDF)",
    data=pdf,
    file_name="medical_report.pdf",
    mime="application/pdf")

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
