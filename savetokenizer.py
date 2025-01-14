import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Örnek metin verisi
texts = ["This is an example sentence.", "Another sentence goes here."]

# Tokenizer'ı oluştur
tokenizer = Tokenizer()

# Tokenizer'ı metinlerle eğit
tokenizer.fit_on_texts(texts)

# Tokenizer'ı kaydetmek için pickle kullan
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Tokenizer başarıyla kaydedildi!")
