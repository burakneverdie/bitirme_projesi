import tkinter as tk
from tkinter import messagebox, scrolledtext
from PIL import Image, ImageTk
import requests
from bs4 import BeautifulSoup
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import threading
import time
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

background_path = "C:/Users/burakk/PycharmProjects/bitirme-projesi/bitirme/yapayzekaresmi.jpg"

model_path = "C:/Users/burakk/PycharmProjects/bitirme-projesi/model/sentiment_model.keras"
tokenizer_path = "C:/Users/burakk/PycharmProjects/bitirme-projesi/bitirme/tokenizer.pickle"

max_length = 100
stopwords_set = set(stopwords.words('turkish'))

with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

model = load_model(model_path)

#veriyi ön işleme
def preprocess_text(text):
    text = text.lower()
    text = re.sub("[^abcçdefgğhıijklmnoöprsştuüvyz]", " ", text)
    text = word_tokenize(text)
    text = [word for word in text if word not in stopwords_set]
    return text

def analyze_comments(comments):
    results = []
    for comment in comments:
        processed_input = preprocess_text(comment)
        input_seq = pad_sequences([tokenizer.texts_to_sequences([processed_input])[0]], maxlen=max_length, padding='post')
        prediction = model.predict(input_seq)
        predicted_label = np.argmax(prediction)
        probabilities = prediction[0] * 100
        results.append((comment, predicted_label, probabilities))
    return results

#web scraping
def fetch_data_from_website(url):
    try:
        header = {
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=header)
        soup = BeautifulSoup(response.content, 'html.parser')
        comments = soup.find_all('p')
        return [comment.get_text() for comment in comments]
    except Exception as e:
        messagebox.showerror("Hata", f"Web sitesinden veri çekilemedi.\nHata: {e}")
        return []

#dosyaya kaydetme fonksiyonu
def save_comments_to_txt(comments, filename="comments.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        for comment in comments:
            file.write(comment + "\n")

#yükleme animasyonu fonksiyonları
loading_frame = None  # Modül seviyesinde tanımlandı
def start_loading_animation():
    global loading_frame
    loading_frame = tk.Frame(pencere, bg="white")
    loading_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    loading_canvas = tk.Canvas(loading_frame, width=100, height=100, bg="white", highlightthickness=0)
    loading_canvas.pack()
    rotate_loading(loading_canvas)

    loading_label = tk.Label(loading_frame, text="Lütfen Bekleyin...", bg="white", font=("Arial", 12), fg="blue")
    loading_label.pack(pady=10)

def stop_loading_animation():
    if loading_frame:
        loading_frame.destroy()

def rotate_loading(canvas, angle=0):
    x1, y1, x2, y2 = 25, 25, 75, 75
    canvas.delete("all")
    canvas.create_arc(
        x1, y1, x2, y2, start=angle, extent=270, style=tk.ARC, outline="blue", width=5
    )
    new_angle = (angle + 10) % 360
    if canvas.winfo_exists():
        canvas.after(50, rotate_loading, canvas, new_angle)

#url analiz fonksiyonu
def analyze_url():
    def process_analysis():
        start_loading_animation()
        time.sleep(2)

        url = url_entry.get()
        if not url:
            messagebox.showwarning("Uyarı", "Lütfen bir URL girin!")
            stop_loading_animation()
            return

        comments = fetch_data_from_website(url)
        if not comments:
            stop_loading_animation()
            return

        #txt dosyasına kaydetme
        save_comments_to_txt(comments)

        #kaydedilen veriyi okuyup duygu analizi yapma
        with open("comments.txt", "r", encoding="utf-8") as file:
            comments_from_file = file.readlines()

        results = analyze_comments(comments_from_file)
        result_text.delete(1.0, tk.END)
        for comment, label, probabilities in results:
            result_text.insert(tk.END, f"Yorum: {comment}\n")
            result_text.insert(tk.END, f"Tahmin: {'Pozitif' if label == 2 else 'Negatif' if label == 0 else 'Nötr'}\n")
            result_text.insert(tk.END, "Sınıflar için olasılıklar:\n")
            result_text.insert(tk.END, f"  Negatif: %{probabilities[0]:.2f}\n")
            result_text.insert(tk.END, f"  Nötr: %{probabilities[1]:.2f}\n")
            result_text.insert(tk.END, f"  Pozitif: %{probabilities[2]:.2f}\n")
            result_text.insert(tk.END, "-" * 50 + "\n")

        stop_loading_animation()

    threading.Thread(target=process_analysis).start()

#tkinter arayüzü
pencere = tk.Tk()
pencere.title("URL Sentiment Analizi")
pencere.geometry("800x600")

def set_background():
    background_image = Image.open(background_path)
    background_image = background_image.resize((800, 600), Image.Resampling.LANCZOS)
    background_photo = ImageTk.PhotoImage(background_image)
    background_label = tk.Label(pencere, image=background_photo)
    background_label.image = background_photo
    background_label.place(relwidth=1, relheight=1)

set_background()

def update_producer_label(_=None):
    x_position = pencere.winfo_width() - producer_label.winfo_width() - 20
    y_position = pencere.winfo_height() - producer_label.winfo_height() - 10
    producer_label.place(x=x_position, y=y_position)

producer_label = tk.Label(
    pencere,
    text="Yapımcılar: Kubilay Ata ve Burak Reis Çıbık",
    font=("Arial", 10),
    bg="white"
)
producer_label.place(x=0, y=0)

pencere.bind("<Configure>", update_producer_label)

url_label = tk.Label(pencere, text="URL Girin:", font=("Arial", 12), bg="white")
url_label.pack(pady=5)

url_entry = tk.Entry(pencere, width=50)
url_entry.pack(pady=5)

analyze_button = tk.Button(pencere, text="Analiz Et", command=analyze_url)
analyze_button.pack(pady=10)

result_text = scrolledtext.ScrolledText(pencere, height=20, width=80)
result_text.pack(pady=10)

pencere.mainloop()
