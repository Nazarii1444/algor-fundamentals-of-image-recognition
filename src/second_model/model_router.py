from io import BytesIO

import numpy as np
from PIL import Image
from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.preprocessing import image

second_model_router = APIRouter()

def prompt(food, question):
    return \
    f""" You are a helpful assistant in my food classification project.
    Try to answer the question about a dish as best as you can, but you can provide only 1-2 sentences.
    You must always answer the question, don't say "I don't know" or "I can't answer that".
    Output should only contain the answer, without any additional text, only on english!
    
    That's a food: {food}
    That's the question: {question}
    """

@second_model_router.post("/answer")
async def predict(request: Request, file: UploadFile = File(...), question: str = None):
    model = request.app.state.model
    classes = request.app.state.classes
    llm = request.app.state.llm

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    if question:
        return JSONResponse(content={
            "predicted_class": classes[int(predicted_class)],
            "message": llm.generate_answer(prompt(question, predicted_class)),
        })
    else:
        return JSONResponse(content={
            "predicted_class": classes[int(predicted_class)],
        })


@second_model_router.post("/predict")
async def answer(request: Request, file: UploadFile = File(...)):
    model = request.app.state.model
    classes = request.app.state.classes

    contents = await file.read()
    img = Image.open(BytesIO(contents)).convert("RGB")

    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    return JSONResponse(content={
        "predicted_class": classes[int(predicted_class)],
    })