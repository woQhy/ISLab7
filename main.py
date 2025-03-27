import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import spacy

# Загружаем модель SpaCy для русского языка
nlp = spacy.load("ru_core_news_sm")

# 1. Загрузка данных
def load_data(data_dir):
    texts = []
    labels = []
    for author in os.listdir(data_dir):
        author_dir = os.path.join(data_dir, author)
        if os.path.isdir(author_dir):
            for file in os.listdir(author_dir):
                if file.endswith(".txt"):
                    try:
                        with open(os.path.join(author_dir, file), "r", encoding="utf-8") as f:
                            texts.append(f.read())
                            labels.append(author)
                    except UnicodeDecodeError:
                        with open(os.path.join(author_dir, file), "r", encoding="cp1251") as f:
                            texts.append(f.read())
                            labels.append(author)
    return pd.DataFrame({"text": texts, "label": labels})

data = load_data("D:\\VS project\\IS\\lab7\\data")  # Папка с подпапками авторов

# 2. Фильтрация классов с недостаточным количеством примеров
min_samples = 6  # Минимум 6 примеров на класс
label_counts = data['label'].value_counts()
valid_labels = label_counts[label_counts >= min_samples].index
data = data[data['label'].isin(valid_labels)]

# Проверка распределения классов
print("Распределение классов после фильтрации:")
print(data['label'].value_counts())

# 3. Разделение данных с сохранением распределения классов
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.2,
    random_state=42,
    stratify=data["label"]
)

# 4. Предобработка текста
def preprocess(text):
    # Удаление знаков препинания и приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

X_train_clean = X_train.apply(preprocess)
X_test_clean = X_test.apply(preprocess)

# 5. Кодирование меток
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# 6. Обучение моделей
# Метрики для сравнения
results = {}

# 6.1 BoW + Logistic Regression
vectorizer = TfidfVectorizer(max_features=1000)  # Создаём TF-IDF векторизатор (максимум 1000 слов)
X_train_bow = vectorizer.fit_transform(X_train_clean)  # Преобразуем обучающие тексты в числовой формат
X_test_bow = vectorizer.transform(X_test_clean)  # Преобразуем тестовые тексты в числовой формат

model_bow = LogisticRegression(max_iter=1000)  # Создаём модель логистической регрессии
model_bow.fit(X_train_bow, y_train_enc)  # Обучаем модель на обучающих данных
results["BoW"] = accuracy_score(y_test_enc, model_bow.predict(X_test_bow))  # Оцениваем точность на тестовых данных

# 6.2 Word Embeddings + Random Forest
# Обучаем Word2Vec модель
sentences = [text.split() for text in X_train_clean]  # Разбиваем тексты на слова
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  # Обучаем Word2Vec

def text_to_vector(text):
    words = text.split()  # Разбиваем текст на слова
    vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]  # Получаем векторы слов
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Усредняем векторы

X_train_w2v = np.array([text_to_vector(text) for text in X_train_clean])  # Преобразуем обучающие тексты в векторы
X_test_w2v = np.array([text_to_vector(text) for text in X_test_clean])  # Преобразуем тестовые тексты в векторы

model_rf = RandomForestClassifier(n_estimators=100, random_state=42)  # Создаём модель случайного леса
model_rf.fit(X_train_w2v, y_train_enc)  # Обучаем модель на обучающих данных
results["Word2Vec + RF"] = accuracy_score(y_test_enc, model_rf.predict(X_test_w2v))  # Оцениваем точность на тестовых данных

# 7. Вывод результатов
print("\nРезультаты классификации:")
for method, acc in results.items():
    print(f"{method}: {acc:.2f}")

# 8. Обработка входного текста
def process_text(file_path):
    with open(file_path, "r", encoding="cp1251") as f:
        text = f.read()

    # Удаление знаков препинания и приведение к нижнему регистру
    text = re.sub(r'[^\w\s]', '', text.lower())

    # Разбиваем текст на части, чтобы обработать их по отдельности
    max_length = 50000  # Уменьшаем максимальную длину для ускорения обработки
    text_parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]

    lemmas = []
    tokens = []
    filtered = []
    entities = []

    # Обрабатываем только первые несколько частей для ускорения
    for part in text_parts[:5]:  # Ограничиваем количество частей
        doc = nlp(part)
        lemmas.extend([token.lemma_ for token in doc if not token.is_stop])
        tokens.extend([token.text for token in doc])
        filtered.extend([token.text for token in doc if not token.is_stop])
        entities.extend([(ent.text, ent.label_) for ent in doc.ents])

    # Облако слов
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(filtered))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

    return {
        "Леммы": lemmas[:10],  # Первые 10 для примера
        "Токены": tokens[:10],
        "Без стоп-слов": filtered[:10],
        "Сущности": entities[:10],
    }

# Пример использования
input_file = "D:\\VS project\\IS\\lab7\\Булгаков2.txt"
if os.path.exists(input_file):
    processed = process_text(input_file)
    print("\nРезультаты обработки текста:")
    for key, value in processed.items():
        print(f"\n{key}:")
        print(value[:10] if isinstance(value, list) else value)
else:
    print(f"Файл {input_file} не найден")
