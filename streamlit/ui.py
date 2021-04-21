import io

import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st
import json

# interact with FastAPI endpoint
backend = "http://fastapi:8000/predict"


def process(image, server_url: str):

    m = MultipartEncoder(fields={"file": ("filename", image, "image/jpeg")})

    r = requests.post(
        server_url, data=m, headers={"Content-Type": m.content_type}, timeout=8000
    )

    return r


# construct UI layout
st.title("SimCLR OCT Disease Classifier")

st.write(
    """Identify whether an OCT shows signs of CNV, DRUSEN, or DME & obtain an image GradCAM heatmap from the model."""
)  # description and instructions

input_image = st.file_uploader("insert image")  # image upload widget

if st.button("Analyze OCT scan"):

    col1, col2 = st.beta_columns(2)

    if input_image:
        outputs = process(input_image, backend)
        original_image = Image.open(input_image).convert("RGB")
        col1.header("Original")
        col1.image(original_image, use_column_width=True)
        col2.json(json.loads(outputs.content))

    else:
        # handle case with no image
        st.write("Insert an image!")
