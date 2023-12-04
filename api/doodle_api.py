from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
import base64
import io
import numpy as np
import tensorflow as tf
from fastapi.middleware.cors import CORSMiddleware
from quickdraw import QuickDrawData
#####
# uvicorn api.doodle_api:app --reload
####
app = FastAPI()
modelDraw= tf.keras.models.load_model('/modelBaseLine.keras')

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
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    image_tensor = tf.image.resize(image_tensor, [64, 64])
    predictionDraw = modelDraw.predict(image_tensor)
    label_list = QuickDrawData().drawing_names

    first_draw = label_list[np.argmax(predictionDraw)]
    first_prob= float(np.max(predictionDraw))


    second_draw = label_list[np.argsort(np.argmax(predictionDraw, axis = 0))[-2]]
    second_prob = float(np.partition(predictionDraw.flatten(), -2)[-2])


    third_draw = label_list[np.argsort(np.argmax(predictionDraw, axis = 0))[-3]]
    third_prob = float(np.partition(predictionDraw.flatten(), -3)[-3])


    result_dict = {
                first_draw:first_prob ,
                second_draw: second_prob,
                third_draw:third_prob
                   }
    return   result_dict
#####
