import streamlit as st
import numpy as np
from PIL import Image
import os
import tensorflow as tf

st.title("🧠 Brain Tumor Detection AI")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_model.tflite")

st.write("📁 Files:", os.listdir("."))

if not os.path.exists(MODEL_PATH):
    st.error("Model not found")
    st.stop()

# Load TFLite using TensorFlow (NOT tflite_runtime)
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

upload = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

classes = ["glioma", "meningioma", "no tumor", "pituitary"]

if upload:
    img = Image.open(upload).convert("RGB")
    st.image(img, use_container_width=True)

    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]["index"], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])
    prediction = np.argmax(output)

    st.success(f"Prediction: {classes[prediction]}")
