import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Muat model
model_path = os.path.join("model", "model_klasifikasi_pohon.h5")
model = load_model(model_path)

# Muat label kelas dari file
label_map_path = os.path.join("model", "label_map.json")
with open(label_map_path, "r") as f:
    index_to_label = json.load(f)

# Path ke folder dataset
dataset_path = os.path.join("C:\\Users\\thejo\\OneDrive\\Desktop\\klasifikasi_pohon", "dataset")

# Tampilkan folder pohon
folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
if not folders:
    print("❌ Tidak ada folder pohon ditemukan di dalam folder dataset.")
    exit()

print("\n📁 Daftar folder pohon:")
for i, folder in enumerate(folders):
    print(f"{i+1}. {folder}")

# Pilih folder pohon
try:
    folder_index = int(input("\nPilih nomor folder pohon: ")) - 1
    folder_pilihan = folders[folder_index]
except (ValueError, IndexError):
    print("❌ Pilihan tidak valid.")
    exit()

folder_path = os.path.join(dataset_path, folder_pilihan)

# Tampilkan gambar di dalam folder
gambar_list = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
if not gambar_list:
    print("❌ Tidak ada gambar ditemukan di folder tersebut.")
    exit()

print(f"\n🖼️ Daftar gambar di folder '{folder_pilihan}':")
for i, file in enumerate(gambar_list):
    print(f"{i+1}. {file}")

# Pilih gambar
try:
    gambar_index = int(input("\nPilih nomor gambar: ")) - 1
    nama_file = gambar_list[gambar_index]
except (ValueError, IndexError):
    print("❌ Pilihan gambar tidak valid.")
    exit()

# Path lengkap ke gambar
image_path = os.path.join(folder_path, nama_file)

# Preprocessing gambar
img = image.load_img(image_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalisasi

# Prediksi
prediction = model.predict(img_array)[0]
predicted_class_index = np.argmax(prediction)
confidence = prediction[predicted_class_index] * 100
predicted_label = index_to_label[str(predicted_class_index)]

# Hasil
print(f"\n✅ Prediksi: {predicted_label} ({confidence:.2f}%)")
