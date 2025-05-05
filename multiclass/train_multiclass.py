import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

# === Шаг 1: Загрузка ===
df = pd.read_csv("C:\\Users\\abume\\.vscode\\codes\\mlfinal\\mlfinal\\data\\train_clean.csv")


# Удаляем строки без меток (все нули)
label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
df = df[df[label_cols].sum(axis=1) > 0]

# === Шаг 2: Преобразование в один класс ===
df['multiclass_label'] = df[label_cols].idxmax(axis=1)

# === Шаг 3: Урезаем 'toxic' до 500 строк ===
# 1. Урезаем 'toxic'
df_major = df[df['multiclass_label'] == 'toxic'].sample(n=500, random_state=42)

# 2. Oversampling 'threat' до 300 строк
df_threat = df[df['multiclass_label'] == 'threat']
df_threat_oversampled = df_threat.sample(n=300, replace=True, random_state=42)

# 3. Все остальные классы (кроме toxic и threat)
df_other = df[df['multiclass_label'].isin(['severe_toxic', 'obscene', 'insult', 'identity_hate'])]

# 4. Объединяем всё
df_balanced = pd.concat([df_major, df_threat_oversampled, df_other])


# === Шаг 4: Подготовка данных ===
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df_balanced['comment_text'].fillna(""))

le = LabelEncoder()
y = le.fit_transform(df_balanced['multiclass_label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Шаг 5: Обучение ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === Шаг 6: Оценка ===
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Шаг 7: Сохранение модели ===
joblib.dump(model, "multiclass_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
