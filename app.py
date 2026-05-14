import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model("brain_tumor_model.keras", compile=False)

st.title("🧠 Brain Tumor Detection AI")

upload = st.file_uploader("Upload MRI Image", type=["jpg","png","jpeg"])

if upload is not None:
    img = Image.open(upload).convert("RGB")
    st.image(img, use_container_width=True)

    img = img.resize((128,128))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    classes = ["glioma", "meningioma", "no tumor", "pituitary"]
    result = classes[np.argmax(prediction)]

    st.success("Prediction: " + result)