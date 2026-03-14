import requests


nutrition_db = {
    "pizza": {
        "calories": 266,
        "protein": 11,
        "fat": 10,
        "carbs": 33
    },
    "hamburger": {
        "calories": 295,
        "protein": 17,
        "fat": 14,
        "carbs": 24
    },
    "ice_cream": {
        "calories": 207,
        "protein": 3.5,
        "fat": 11,
        "carbs": 24
    }
}


def get_nutrition(food_name: str):
    food_name = food_name.lower()

    if food_name in nutrition_db:
        return nutrition_db[food_name]

    return {
        "calories": None,
        "protein": None,
        "fat": None,
        "carbs": None
    }


def analyze_health(nutrition):
    calories = nutrition.get("calories")
    fat = nutrition.get("fat")
    protein = nutrition.get("protein")

    health_score = "unknown"
    recommendation = ""

    if calories is None:
        return {
            "health_score": "unknown",
            "recommendation": "No nutrition data available."
        }

    if calories < 150:
        health_score = "healthy"
        recommendation = "Low calorie food, suitable for regular consumption."

    elif calories < 300:
        health_score = "moderate"
        recommendation = "Moderate calorie food, eat in balanced portions."

    else:
        health_score = "high-calorie"
        recommendation = "High calorie food, recommended to eat occasionally."

    return {
        "health_score": health_score,
        "recommendation": recommendation
    }


def get_nutrition_from_api(food_name: str):
    url = "https://world.openfoodfacts.org/cgi/search.pl"

    params = {
        "search_terms": food_name,
        "search_simple": 1,
        "action": "process",
        "json": 1
    }

    r = requests.get(url, params=params)
    data = r.json()

    if data["count"] == 0:
        return None

    product = data["products"][0]
    nutriments = product.get("nutriments", {})

    return {
        "calories": nutriments.get("energy-kcal_100g"),
        "fat": nutriments.get("fat_100g"),
        "carbs": nutriments.get("carbohydrates_100g"),
        "protein": nutriments.get("proteins_100g")
    }
