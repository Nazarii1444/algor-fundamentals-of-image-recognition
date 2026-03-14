from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

token = "token"
model_name = "nateraw/food"

processor = AutoImageProcessor.from_pretrained(model_name, token=token)
model = AutoModelForImageClassification.from_pretrained(model_name, token=token)

model.eval()

def predict_food(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")

    with torch.inference_mode():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    label = model.config.id2label[predicted_class_idx]

    return label
