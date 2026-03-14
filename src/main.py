import uvicorn
from fastapi import FastAPI

from src.second_model.llm import LLM
from tensorflow.keras.models import load_model
from src.first_model.router import first_model_router
from src.second_model.model_router import second_model_router
from src.smart_analyze.router import smart_analyze_router

app = FastAPI()
app.include_router(first_model_router, prefix="/first_model", tags=["first_model"])
app.include_router(second_model_router, prefix="/second_model", tags=["second_model"])
app.include_router(smart_analyze_router, prefix="/analysis", tags=["analysis"])

model = load_model('second_model/model_en22.h5')
with open('second_model/class_labels.txt', 'r') as f:
    classes = [line.strip() for line in f]

app.state.model = model
app.state.classes = classes

llm = LLM("token")
app.state.llm = llm


@app.get("/")
async def health_check():
    return {"status": "OK"}


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)
