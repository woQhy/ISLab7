import os
import re
import spacy
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Загрузка модели spaCy для русского языка
nlp = spacy.load("ru_core_news_sm")
nlp.max_length = 2000000  # Увеличиваем лимит длины текста до 2 000 000 символов

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

data = load_data("D:\VS project\IS\lab7\data")  # Папка с подпапками авторов

# Проверка загруженных данных
if data.empty:
    print("Не удалось загрузить данные. Проверьте папку 'data'.")
    exit()

# Проверка распределения классов до фильтрации
print("Распределение классов до фильтрации:")
print(data['label'].value_counts())

# 2. Фильтрация классов с недостаточным количеством примеров
min_samples = 6  # Установите подходящее значение
label_counts = data['label'].value_counts()
valid_labels = label_counts[label_counts >= min_samples].index
data = data[data['label'].isin(valid_labels)]

# Проверка распределения классов после фильтрации
print("Распределение классов после фильтрации:")
print(data['label'].value_counts())

if data.empty:
    print("Нет доступных данных для обучения после фильтрации.")
    exit()

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
    max_length = 1000000
    if len(text) > max_length:
        docs = []
        for i in range(0, len(text), max_length):
            chunk = text[i:i + max_length]
            doc = nlp(chunk, disable=["parser", "ner"])
            docs.append(doc)
        return " ".join([token.lemma_ for doc in docs for token in doc if not token.is_stop and not token.is_punct])
    else:
        doc = nlp(text, disable=["parser", "ner"])
        return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

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
model_bow.fit(X_train_bow, y_train)  # Обучаем модель на обучающих данных
results["BoW"] = accuracy_score(y_test, model_bow.predict(X_test_bow))  # Оцениваем точность на тестовых данных

# 6.2 Word Embeddings
tokenizer = Tokenizer(num_words=10000)  # Создаём токенизатор (максимум 10 000 слов)
tokenizer.fit_on_texts(X_train_clean)  # Обучаем токенизатор на обучающих текстах

X_train_seq = tokenizer.texts_to_sequences(X_train_clean)  # Преобразуем тексты в последовательности чисел
X_test_seq = tokenizer.texts_to_sequences(X_test_clean)  # Преобразуем тестовые тексты в последовательности чисел

max_len = 100  # Максимальная длина последовательности
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)  # Дополняем последовательности до длины 100
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)  # Дополняем тестовые последовательности до длины 100

# 6.3 LSTM
model_lstm = Sequential([
    Embedding(10000, 128, input_length=max_len),  # Слой для преобразования слов в векторы
    LSTM(128),  # Рекуррентный слой LSTM
    Dense(len(data["label"].unique()), activation="softmax")  # Выходной слой для классификации
])
model_lstm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Компиляция модели
model_lstm.fit(X_train_pad, y_train_enc, epochs=5, validation_data=(X_test_pad, y_test_enc), verbose=0)  # Обучение модели
results["LSTM"] = model_lstm.evaluate(X_test_pad, y_test_enc)[1]  # Оценка точности на тестовых данных

# 6.4 Conv1D (1D Convolutional Neural Network) Свёрточная нейронная сеть
model_conv = Sequential([
    Embedding(10000, 128, input_length=max_len),  # Слой для преобразования слов в векторы
    Conv1D(128, 5, activation="relu"),  # Свёрточный слой
    GlobalMaxPooling1D(),  # Слой для извлечения наиболее важных признаков
    Dense(len(data["label"].unique()), activation="softmax")  # Выходной слой для классификации
])
model_conv.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])  # Компиляция модели
model_conv.fit(X_train_pad, y_train_enc, epochs=5, validation_data=(X_test_pad, y_test_enc), verbose=0)  # Обучение модели
results["Conv1D"] = model_conv.evaluate(X_test_pad, y_test_enc)[1]  # Оценка точности на тестовых данных

# 6.5 Word2Vec
sentences = [text.split() for text in X_train_clean]  # Разбиваем тексты на слова
model_w2v = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)  # Обучаем Word2Vec

def text_to_vector(text):
    words = text.split()  # Разбиваем текст на слова
    vectors = [model_w2v.wv[word] for word in words if word in model_w2v.wv]  # Получаем векторы слов
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)  # Усредняем векторы

X_train_w2v = np.array([text_to_vector(text) for text in X_train_clean])  # Преобразуем обучающие тексты в векторы
X_test_w2v = np.array([text_to_vector(text) for text in X_test_clean])  # Преобразуем тестовые тексты в векторы

model_w2v_clf = LogisticRegression(max_iter=1000)  # Создаём модель логистической регрессии
model_w2v_clf.fit(X_train_w2v, y_train)  # Обучаем модель на обучающих данных
results["Word2Vec"] = accuracy_score(y_test, model_w2v_clf.predict(X_test_w2v))  # Оцениваем точность на тестовых данных

# 7. Вывод результатов
print("\nРезультаты классификации:")
for method, acc in results.items():
    print(f"{method}: {acc:.2f}")

# 8. Обработка входного текста
def process_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    doc = nlp(text)
    
    # 8.1 Лемматизация
    lemmas = [token.lemma_ for token in doc]
    
    # 8.2 Токенизация
    tokens = [token.text for token in doc]
    
    # 8.3 Удаление стоп-слов
    filtered = [token.text for token in doc if not token.is_stop]
    
    # 8.4 Извлечение сущностей
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    # 8.5 Облако слов
    wordcloud = WordCloud(width=800, height=400).generate(" ".join(filtered))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()
    
    return {
        "Леммы": lemmas[:10],  # Первые 10 для примера
        "Токены": tokens[:10],
        "Без стоп-слов": filtered[:10],
        "Сущности": entities,
    }

# Пример использования
input_file = "Булгаков2.txt"
if os.path.exists(input_file):
    processed = process_text(input_file)
    print("\nРезультаты обработки текста:")
    for key, value in processed.items():
        print(f"\n{key}:")
        print(value[:10] if isinstance(value, list) else value)
else:
    print(f"Файл {input_file} не найден")