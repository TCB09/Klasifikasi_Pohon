import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, callbacks

# Konfigurasi
img_height, img_width = 150, 150
dataset_path = 'dataset'
batch_size = 8
epochs = 20
val_split = 0.3
model_folder = 'model'

# Data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=val_split,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

# Generator training & validasi
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

num_classes = train_generator.num_classes

# Membangun model CNN
model = models.Sequential([
    tf.keras.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback
os.makedirs(model_folder, exist_ok=True)
checkpoint_cb = callbacks.ModelCheckpoint(
    filepath=os.path.join(model_folder, "model_klasifikasi_pohon.h5"),
    save_best_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1
)

earlystop_cb = callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

# Train model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint_cb, earlystop_cb]
)

print("✅ Model terbaik disimpan ke folder 'model'")

# Simpan label
label_map = train_generator.class_indices
index_to_label = {v: k for k, v in label_map.items()}
with open(os.path.join(model_folder, "label_map.json"), "w") as f:
    json.dump(index_to_label, f)

print("✅ Label kelas disimpan ke 'model/label_map.json'")
