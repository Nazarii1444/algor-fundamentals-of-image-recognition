import io

from PIL import Image
from fastapi import UploadFile, File, APIRouter
from src.first_model.model import predict_food
from src.first_model.nutrition import get_nutrition

first_model_router = APIRouter()


@first_model_router.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    food_name = predict_food(image)
    nutrition = get_nutrition(food_name)

    return {
        "food": food_name,
        "nutrition": nutrition
    }
