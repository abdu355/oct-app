import io

from predict import get_classifier, get_prediction,get_gradcam
from starlette.responses import Response
from fastapi import FastAPI, File
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

model = get_classifier()

app = FastAPI(
    title="SimCLR OCT Disease Classifier",
    description="""Identify whether an OCT shows signs of CNV, DRUSEN, or DME & obtain an image GradCAM heatmap from the model.""",
    version="1.0.0",
)


@app.post("/predict")
def predict(file: bytes = File(...)):
    """Get prediction"""  
    class_id, class_name = get_prediction(image_bytes=file)  
    data = {'class_id': class_id, 'class_name': class_name}
    res = jsonable_encoder(data)
    return JSONResponse(content=res)

@app.post("/gradcam")
def gradcam(file: bytes = File(...)):
    """Get gradcam"""  
    img_gradcam = get_gradcam(image_bytes=file) 
    bytes_io = io.BytesIO()
    img_gradcam.save(bytes_io, format="PNG")
    return Response(bytes_io.getvalue(), media_type="image/png")