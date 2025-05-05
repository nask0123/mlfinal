from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Загрузка
model = joblib.load("multilabel_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    y_pred = model.predict(X)[0]
    
    result = {label: int(pred) for label, pred in zip(label_cols, y_pred)}
    return result
