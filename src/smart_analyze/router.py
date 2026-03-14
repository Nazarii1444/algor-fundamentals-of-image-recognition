import io
import numpy as np
from fastapi import UploadFile, File, APIRouter, Request, Query
from PIL import Image
from tensorflow.keras.preprocessing import image as keras_image

from src.first_model.model import predict_food
from src.first_model.nutrition import get_nutrition, analyze_health, get_nutrition_from_api

smart_analyze_router = APIRouter()

@smart_analyze_router.post("/smart-analyze")
async def smart_analyze(
        request: Request,
        file: UploadFile = File(...),
        model_type: str = Query("hf")
):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")

    # 1. Food recognition
    if model_type == "tf":
        tf_model = request.app.state.model
        classes = request.app.state.classes

        img_resized = img.resize((224, 224))
        img_array = keras_image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)

        predictions = tf_model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        food = classes[int(predicted_class)]

    else:
        # HuggingFace model
        food = predict_food(img)

    # 2. Nutrition lookup
    nutrition = get_nutrition_from_api(food)

    # 3. Health analysis
    health = analyze_health(nutrition)

    # 4. AI explanation
    llm = request.app.state.llm

    prompt = f"""
    Food: {food}

    Calories: {nutrition.get("calories")}
    Protein: {nutrition.get("protein")}
    Fat: {nutrition.get("fat")}
    Carbs: {nutrition.get("carbs")}

    Health score: {health.get("health_score")}

    Explain briefly:
    - Is this food healthy?
    - When should people eat it?
    """

    ai_explanation = llm.generate_answer(prompt)

    return {
        "food": food,
        "nutrition": nutrition,
        "health_analysis": health,
        "ai_explanation": ai_explanation
    }
