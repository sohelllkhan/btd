import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

# Load TFLite model
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

st.title("🧠 Brain Tumor Detection AI")

upload = st.file_uploader("Upload MRI Image", type=["jpg", "png", "jpeg"])

classes = ["glioma", "meningioma", "no tumor", "pituitary"]

if upload is not None:
    img = Image.open(upload).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img = img.resize((128, 128))
    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output)

    st.success("Prediction: " + classes[class_index])
