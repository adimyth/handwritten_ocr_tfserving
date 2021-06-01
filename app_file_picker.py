import os
import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import requests
import streamlit as st  # type: ignore
from PIL import Image  # type: ignore
from streamlit_drawable_canvas import st_canvas  # type: ignore

from predictor import Predictor  # type: ignore

if __name__ == "__main__":
    predictor = Predictor()

    # file uploader
    st.title("Handwritten OCR - TensorFlow Serving")
    uploaded_file = st.file_uploader(label="Upload Image", type=["jpg", "png"])

    # sidebar
    version = st.sidebar.selectbox("Select Version", ("Version 1", "Version 2"))
    endpoint = f"http://localhost:8501/v1/models/handwritten_ocr/versions/{version.split()[1]}:predict"

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"{uploaded_file.name} - {img.size}")

    if st.button("Predict"):
        if uploaded_file is None:
            st.error("Select an image")
        else:
            # save image temporarily in RGB
            temp_path = "temp.png"
            Image.open(uploaded_file).convert("RGB").save(temp_path)
            input_data = np.expand_dims(
                predictor.encode_single_sample(temp_path), axis=0
            )

            # Prepare the data that is going to be sent in the POST request
            json_data = json.dumps({"instances": input_data.tolist()})
            headers = {"content-type": "application/json"}
            # Send the request to the Prediction API
            response = requests.post(endpoint, data=json_data, headers=headers)

            interim = np.array(response.json()["predictions"][0])
            prediction = predictor.decode_predictions(np.expand_dims(interim, axis=0))[
                0
            ]
            prediction = prediction.replace("[UNK]", "").replace("]", "")
            st.success(f"Prediction: {prediction}")
        os.remove(temp_path)
