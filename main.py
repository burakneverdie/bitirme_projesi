import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

stopwords = set(stopwords.words('turkish'))

# Veri seti ve test setini yükleme
train_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/train.csv')
test_df = pd.read_csv('C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/test.csv')

# Metindeki gereksiz karakterleri çıkarma ve sadece Türkçe harfleri bırakma
def preprocess_text(text):
    text = text.lower()  # Metni küçük harfe çevir
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]", " ", text)
    text = word_tokenize(text)  # Cümledeki kelimeleri ayır
    text = [word for word in text if word not in stopwords]  # Stopwords'leri kaldır
    text = " ".join(text)
    return text

# Verinin ön işlenmesi
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

# Eğitim ve doğrulama setini ayırma
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train_encoded, test_size=0.2, random_state=42)

# Pipeline
model_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('clf', LogisticRegression(max_iter=500))
])

# Modeli eğitme
model_pipeline.fit(X_train, y_train)

# Modelin doğrulama seti ile test edilmesi
val_accuracy = model_pipeline.score(X_val, y_val)
print(f'doğruluk oranı: {val_accuracy:.2f}')

# Test setinde test etme
y_pred = model_pipeline.predict(X_test)

# Tahminleri geri döndürme
y_pred_labels = label_encoder.inverse_transform(y_pred)  # Sayısal etiketleri tekrardan duygulara çevirme

# Sonuçların değerlendirilmesi
report = classification_report(test_df['duygu'], y_pred_labels, target_names=label_encoder.classes_, output_dict=True)

# `classification_report` çıktısındaki sınıf anahtarlarını güvenli şekilde kullanarak yazdırma
for label in label_encoder.classes_:
    if label in report:
        print(f"{label} sınıfı için:")
        print(f"  Precision: {report[label].get('precision', 0):.2f}")
        print(f"  Recall: {report[label].get('recall', 0):.2f}")
        print(f"  F1-score: {report[label].get('f1-score', 0):.2f}")
        print()
    else:
        print(f"{label} sınıfı raporda mevcut değil.")

# Karışıklık matrisi oluşturma ve görselleştirme
conf_matrix = confusion_matrix(y_test_encoded, y_pred, labels=label_encoder.transform(label_encoder.classes_))
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title('Karışıklık Matrisi')
plt.show()

# Kullanıcıdan metin girdisi alma
while True:
    user_input = input("Bir metin girin: ")
    processed_input = preprocess_text(user_input)
    prediction = model_pipeline.predict([processed_input])
    predicted_label = label_encoder.inverse_transform(prediction)
    print(f'Tahmin edilen duygu: {predicted_label[0]}')

    while True:
        continue_choice = input("Devam etmek istiyor musunuz? (E/H): ").strip().upper()
        if continue_choice in ['E', 'H']:
            break
        else:
            print("Lütfen yalnızca 'E' veya 'H' girin.")

    if continue_choice != 'E':
        print("Program sonlandırıldı.")
        break
