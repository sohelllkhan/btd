import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import gdown
import os

# -------------------------------
# 1. DOWNLOAD MODEL FROM GOOGLE DRIVE (ONLY FIRST RUN)
# -------------------------------

MODEL_PATH = "brain_tumor_model.keras"

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model... please wait")

    url = "https://drive.google.com/file/d/1RLNXwoXA_0tMi36PP8EZmxsTrPAgLO6B/view?usp=drive_link"  # 🔴 replace this
    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

# -------------------------------
# 2. LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -------------------------------
# 3. UI
# -------------------------------
st.title("🧠 Brain Tumor Detection AI")

upload = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

classes = ["glioma", "meningioma", "no tumor", "pituitary"]

if upload is not None:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # preprocess
    img = img.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # prediction
    prediction = model.predict(img)
    class_index = np.argmax(prediction)

    st.success("Prediction: " + classes[class_index])
