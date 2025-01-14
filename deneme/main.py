import pandas as pd # kullanılan kütüphaneler pandas scikit-learn transformers

#veri seti ve test dosyasını yükleme
train_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/train.csv')
test_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/test.csv')

# ilk satırları incelemek için
print(train_df.head())
print(test_df.head())

#karakter ve boşlukları temizleme
def preprocess_text(text):
    text = text.lower()  # küçük harfe çevirir
    text = text.replace('\n', ' ')  # yeni satırları kaldırır
    return text

# ön veri işleme
train_df['yorum'] = train_df['yorum'].apply(preprocess_text)
test_df['yorum'] = test_df['yorum'].apply(preprocess_text)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# duyguları sayısal etikete çevirme
label_encoder = LabelEncoder()
train_df['duygu'] = label_encoder.fit_transform(train_df['duygu'])

# eğitim ve doğrulama setlerine ayırma
X_train, X_val, y_train, y_val = train_test_split(train_df['yorum'], train_df['duygu'], test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

# TF-IDF vektörleştirme
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)

from sklearn.linear_model import LogisticRegression

# model oluşturma ve eğitme
model = LogisticRegression(max_iter=500)
model.fit(X_train_tfidf, y_train)

# doğrulama setinde model test etme
accuracy = model.score(X_val_tfidf, y_val)
print(f'doğruluk oranı: {accuracy:.2f}')

# kullanıcıdan girdi alma
user_input = input("Bir metin girin: ")

# metnin işlenmesi ve vektörleştirme
processed_input = preprocess_text(user_input)
input_tfidf = tfidf.transform([processed_input])

# tahmin yapma
prediction = model.predict(input_tfidf)
predicted_label = label_encoder.inverse_transform(prediction)

print(f'tahmin edilen duygu: {predicted_label[0]}')






