import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

DATASET_DIR = "dataset"
IMG_SIZE = 128

data = []
labels = []

for person in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person)
    for img_name in os.listdir(person_path):
        img_path = os.path.join(person_path, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        data.append(img)
        labels.append(person)

data = np.array(data) / 255.0
data = data.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

le = LabelEncoder()
labels = le.fit_transform(labels)
labels = to_categorical(labels)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(labels.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(data, labels, epochs=10)

model.save("model/face_model.h5")
print("Model berhasil disimpan")
