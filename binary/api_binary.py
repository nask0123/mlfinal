from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Загружаем модель и векторизатор
model = joblib.load("binary_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    vec = vectorizer.transform([input.text])
    pred = model.predict(vec)
    return {"prediction": int(pred[0])}
