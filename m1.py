import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Stopwords'leri ayarla
stopwords_set = set(stopwords.words('turkish'))

# Veri setlerini yükle
train_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/train.csv')
test_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/test.csv')

# Metin ön işleme fonksiyonu
def preprocess_text(text):
    text = text.lower()  # Küçük harf
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]", " ", text)  # Türkçe harfler dışındakileri kaldır
    text = word_tokenize(text)  # Kelimelere böl
    text = [word for word in text if word not in stopwords_set]  # Stopwords kaldır
    return text

# Veriyi temizle
train_df["clean_text"] = train_df["yorum"].apply(preprocess_text)
test_df["clean_text"] = test_df["yorum"].apply(preprocess_text)

# Eğitim ve test seti
X_train = train_df["clean_text"]
X_test = test_df["clean_text"]
y_train = train_df["duygu"]
y_test = test_df["duygu"]

# Etiketleri sayısallaştırma
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Eğitim ve doğrulama setine ayır
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Word2Vec modelini eğit
word2vec_model = Word2Vec(sentences=X_train, vector_size=100, window=5, min_count=1, workers=4)

# Embedding matrisi oluştur
vocab_size = len(word2vec_model.wv) + 1
embedding_matrix = np.zeros((vocab_size, 100))
word_index = {word: i + 1 for i, word in enumerate(word2vec_model.wv.index_to_key)}
for word, i in word_index.items():
    embedding_matrix[i] = word2vec_model.wv[word]

# Metinleri sıralara dönüştür
def texts_to_sequences(texts, word_index):
    sequences = []
    for text in texts:
        sequences.append([word_index.get(word, 0) for word in text])
    return sequences

X_train_seq = texts_to_sequences(X_train, word_index)
X_val_seq = texts_to_sequences(X_val, word_index)
X_test_seq = texts_to_sequences(X_test, word_index)

# Sıraları aynı uzunluğa getirme (padding)
max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_val_pad = pad_sequences(X_val_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Modeli kaydedeceğiniz dosyanın yolu (native Keras formatında)
model_path = "C:/Users/burakk/PycharmProjects/bitirme-projesi/model/sentiment_model.keras"

# Model oluşturma veya yükleme
try:
    model = load_model(model_path)
    print("Eğitilmiş model yüklendi.")
except:
    print("Model bulunamadı. Yeni model oluşturuluyor.")
    # Yeni modeli oluştur
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix], input_length=max_length, trainable=False),
        LSTM(128, return_sequences=False),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(len(label_encoder.classes_), activation='softmax')
    ])

    # Modeli derle
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Modeli eğit
    model.fit(X_train_pad, y_train, validation_data=(X_val_pad, y_val), epochs=10, batch_size=32)

    # Modeli kaydet
    model.save(model_path)  # .keras formatında kaydediyoruz
    print("Model eğitildi ve kaydedildi.")

# Test setinde tahmin yapma
y_pred = model.predict(X_test_pad)
y_pred_labels = np.argmax(y_pred, axis=1)

# Doğruluk oranı
accuracy = (y_pred_labels == y_test_encoded).mean()
print(f"Test Doğruluk Oranı: {accuracy:.2f}")

# Kullanıcıdan metin girdisi alma
while True:
    user_input = input("Bir metin girin: ")
    processed_input = preprocess_text(user_input)
    input_seq = pad_sequences([texts_to_sequences([processed_input], word_index)[0]], maxlen=max_length, padding='post')
    prediction = model.predict(input_seq)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    print(f"Tahmin edilen duygu: {predicted_label[0]}")

    continue_choice = input("Devam etmek istiyor musunuz? (E/H): ").strip().upper()
    if continue_choice != 'E':
        print("Program sonlandırıldı.")
        break
