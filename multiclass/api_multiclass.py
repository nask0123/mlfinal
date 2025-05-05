from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Загрузка моделей
model = joblib.load("multiclass_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    y_pred = model.predict(X)
    label = label_encoder.inverse_transform(y_pred)[0]
    return {"predicted_class": label}
