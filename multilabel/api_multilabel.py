from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

# Load model and vectorizer
model = joblib.load("multilabel_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

app = FastAPI()

# ✅ Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    X = vectorizer.transform([input.text])
    probas = model.predict_proba(X)  # returns list of arrays

    threshold = 0.7
    result = {}

    for i, label in enumerate(label_cols):
        prob_1 = probas[i][:, 1][0]  # probability for class=1
        result[label] = int(prob_1 >= threshold)

    return result
