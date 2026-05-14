import streamlit as st
import numpy as np
from PIL import Image
import os
import tflite_runtime.interpreter as tflite

st.title("🧠 Brain Tumor Detection AI")

# -----------------------------
# Safe model path (IMPORTANT FIX)
# -----------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_model.tflite")

# Debug: show files in deployment
st.write("📁 Files in app directory:", os.listdir("."))

# Check if model exists
if not os.path.exists(MODEL_PATH):
    st.error("❌ Model file NOT found! Make sure 'brain_tumor_model.tflite' is in GitHub repo.")
    st.stop()

# Load TFLite model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# UI
# -----------------------------
upload = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

classes = ["glioma", "meningioma", "no tumor", "pituitary"]

if upload is not None:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output)

    st.success("Prediction: " + classes[class_index])
