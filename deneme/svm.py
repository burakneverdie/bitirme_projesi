import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # SVM modelini import et
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


stopwords = set(stopwords.words('turkish'))

# Veri setini yükle
train_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/train.csv',nrows=1000)
test_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/test.csv',nrows=1000)

# Metin temizleme fonksiyonu
def preprocess_text(text):
    text = text.lower()  # Metni küçük harfe çevir
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]", " ", text)  # Türkçe karakterleri koruyarak diğerlerini kaldır
    text = word_tokenize(text)  # Cümledeki kelimeleri ayır
    text = [word for word in text if word not in stopwords]  # Stopwords'leri kaldır
    text = " ".join(text)
    return text

# Ön veri işleme
train_df["clean_text"] = train_df["yorum"].apply(lambda x: preprocess_text(x))
test_df["clean_text"] = test_df["yorum"].apply(lambda x: preprocess_text(x))

X_train = train_df["clean_text"]
X_test = test_df["clean_text"]
y_train = train_df["duygu"]
y_test = test_df["duygu"]

# Duyguları sayısal etikete çevirme
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Eğitim ve doğrulama setlerine ayırma
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Pipeline oluşturma
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', SVC())  # SVM modelini ekleyin
])

# Modeli eğitme
model_pipeline.fit(X_train, y_train)

# Modeli doğrulama setinde test etme
val_accuracy = model_pipeline.score(X_val, y_val)
print(f'Doğrulama setindeki doğruluk oranı: {val_accuracy:.2f}')

# Test setinde model test etme
y_pred = model_pipeline.predict(X_test)

# Tahminleri geri dönüştür
y_pred_labels = label_encoder.inverse_transform(y_pred)  # Sayısal etiketleri orijinal etiketlere çevir

# Sonuçları değerlendirme
print(classification_report(test_df['duygu'], y_pred_labels, target_names=label_encoder.classes_))

# Kullanıcıdan girdi alma ve devam etme seçeneği
while True:
    user_input = input("Bir metin girin: ")
    processed_input = preprocess_text(user_input)
    prediction = model_pipeline.predict([processed_input])
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f'Tahmin edilen duygu: {predicted_label[0]}')

    # Devam etmek isteyip istemediğini sorma
    continue_choice = input("Devam etmek istiyor musunuz? (E/H): ").strip().upper()
    if continue_choice != 'E':
        print("Program sonlandırıldı.")
        break
