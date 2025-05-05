import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib

# Загрузка обучающих данных
df = pd.read_csv("C:\\Users\\abume\\.vscode\\codes\\mlfinal\\mlfinal\\data\\train_clean.csv")


# Используем только одну метку — 'toxic'
df = df[['comment_text', 'toxic']].dropna()
df['toxic'] = df['toxic'].astype(int)

# TF-IDF векторизация
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['comment_text'])
y = df['toxic']

# Разделим на train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучаем модель
model = MultinomialNB()
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Сохраняем модель и векторизатор
joblib.dump(model, 'binary_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
