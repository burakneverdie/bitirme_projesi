import pandas as pd


# Verileri yükleme
df = pd.read_csv("C:/Users/burakk/PycharmProjects/bitirme-projesi/datasets/train.csv")  # Yolu güncelleyin

# Duygu dağılımını kontrol etme
value_counts = df['duygu'].value_counts()
print("duygu dağılımı:")
print(value_counts)
