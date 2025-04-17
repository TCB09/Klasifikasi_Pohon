# evaluate_model.py
#notif kuning = versi python masih bawaan terbaru

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Konfigurasi
img_height, img_width = 150, 150
batch_size = 8
model_path = os.path.join("model", "model_klasifikasi_pohon.h5")
label_map_path = os.path.join("model", "label_map.json")
dataset_path = "dataset"

# Muat model
model = tf.keras.models.load_model(model_path)

# Muat label
with open(label_map_path, "r") as f:
    index_to_label = json.load(f)

# Generator untuk data validasi (tanpa augmentasi)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # penting agar prediksi sesuai urutan label
)

# Evaluasi akurasi
loss, acc = model.evaluate(val_generator)
print(f"\n✅ Akurasi validasi: {acc * 100:.2f}%")

# Prediksi label
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = val_generator.classes

# Tampilkan classification report
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=val_generator.class_indices.keys())

# Plot confusion matrix
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("📊 Confusion Matrix")
plt.tight_layout()
plt.show()
