import io
import os
import requests
from PIL import Image
from requests_toolbelt.multipart.encoder import MultipartEncoder
import streamlit as st
import json

# interact with FastAPI endpoint
# url = st.secrets["fastapi_url"]

# class_api = f"{url}/predict"
# gradcam_api = f"{url}/gradcam"

class_api = "http://fastapi:8000/predict"
gradcam_api = "http://fastapi:8000/gradcam"

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
    col3, col4 = st.beta_columns(2)

    if input_image:
        outputs_class = process(input_image, class_api)
        outputs_gradcam = process(input_image, gradcam_api)     
        classification_res = json.loads(outputs_class.content)

        original_image = Image.open(input_image).convert("RGB")
        gradcam_image = Image.open(io.BytesIO(outputs_gradcam.content)).convert("RGB")

        col1.header("Original Input")
        col1.image(original_image, use_column_width=True)
        col2.header("Output")
        col2.json(classification_res)
        col3.header("GradCAM")
        col3.image(gradcam_image, use_column_width=True)

    else:
        # handle case with no image
        st.write("Insert an image!")
