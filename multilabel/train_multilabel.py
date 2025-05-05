import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
import joblib

# === Шаг 1: Загрузка данных ===
df = pd.read_csv("C:\\Users\\abume\\.vscode\\codes\\mlfinal\\mlfinal\\data\\train_clean.csv")


label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
# Удаляем пустые строки
# Удаляем пустые строки
df = df[['comment_text'] + label_cols].dropna()

# Собираем по 1000 примеров для каждого класса
dfs = []

for label in label_cols:
    df_label = df[df[label] == 1]
    df_label_balanced = df_label.sample(n=1000, replace=True, random_state=42)
    dfs.append(df_label_balanced)

# Объединяем всё и убираем дубликаты (один текст может быть в нескольких классах)
df_balanced = pd.concat(dfs).drop_duplicates(subset=["comment_text"])
df = df_balanced.reset_index(drop=True)


# === Шаг 2: Преобразование ===
X_texts = df['comment_text'].fillna("")
y = df[label_cols].astype(int)

# === Шаг 3: Векторизация текста ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(X_texts)

# === Шаг 4: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Шаг 5: Обёртка MultiOutput ===
model = MultiOutputClassifier(MultinomialNB())
model.fit(X_train, y_train)

# === Шаг 6: Оценка ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_cols))


# === Шаг 7: Сохранение ===
joblib.dump(model, "multilabel_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
