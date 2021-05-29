import json

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import requests
import streamlit as st  # type: ignore
from PIL import Image  # type: ignore
from streamlit_drawable_canvas import st_canvas  # type: ignore

from predictor import Predictor  # type: ignore
from image_processor import get_splits


if __name__ == "__main__":
    predictor = Predictor()
    endpoint = "http://localhost:8501/v1/models/handwritten_ocr/versions/1:predict"
    # title
    st.title("Handwritten OCR - TensorFlow Serving")
    st.write("Model is trained on a word-level. In case a sentence is passed, it will make prediction per word")

    # sidebar
    technique = st.sidebar.selectbox(
        "Decoding Technique", ("Greedy Search", "Beam Search")
    )
    if technique == "Beam Search":
        beam_width = st.slider("Beam Width", 0, 20, 5)
        top_paths = st.slider("Top K predictions", 0, 5, 1)

    # canvas component
    canvas_result = st_canvas(
        height=250,
        width=600,
        fill_color="#000000",
        stroke_width=5,
        stroke_color="#000000",
        background_color="#FFFFFF",
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict"):
        if canvas_result.image_data is None:
            st.error("Write something")
        else:
            img = canvas_result.image_data.astype("float")
            img = Image.fromarray(img.astype(np.uint8)).convert("RGB")

            # Temporarily saving as RGB
            temp_path = "temp.png"
            plt.imshow(img)
            plt.axis("off")
            plt.savefig(temp_path)

            splits = get_splits(temp_path)
            for idx, split in enumerate(splits):
                st.image(split, caption=split.shape)
                plt.imshow(split)
                plt.axis("off")
                plt.savefig(f"{idx}.png")

                # Converting to desired format
                input_data = np.expand_dims(
                    predictor.encode_single_sample(f"{idx}.png"), axis=0
                )

                # Prepare the data that is going to be sent in the POST request
                json_data = json.dumps({"instances": input_data.tolist()})
                headers = {"content-type": "application/json"}

                # Send the request to the Prediction API
                response = requests.post(endpoint, data=json_data, headers=headers)
                interim = np.array(response.json()["predictions"][0])

                # Decoding based on technique
                if technique == "Greedy Search":
                    prediction = predictor.decode_predictions(
                        np.expand_dims(interim, axis=0)
                    )[0]
                    prediction = prediction.replace("[UNK]", "")
                    st.success(f"Prediction: {prediction}")
                else:
                    predictions = predictor.decode_predictions_beam(
                        np.expand_dims(interim, axis=0), beam_width, top_paths
                    )
                    predictions = [x.replace("[UNK]", "") for x in predictions]
                    for i, pred in enumerate(predictions):
                        st.write(f"{i}: {pred}")
