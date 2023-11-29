from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
import base64
import io
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
#####
# uvicorn api.doodle_api:app --reload
####
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#####
@app.get("/")
def index():
    return {"status": "ok"}
#####

@app.get("/test")
def index():
    return {"status": "ok"}

class Drawing(BaseModel):
    data: str

######
@app.post("/predict")
async def predict(image: Drawing):
    data = image.data
    image_string = data.split(',')[1]
    base64_decoded = base64.b64decode(image_string)
    image_image = Image.open(io.BytesIO(base64_decoded))
    image_np = np.array(image_image)
    image_tensor = tf.convert_to_tensor(image_np)
    return {"prediction": "ok"}
#####
