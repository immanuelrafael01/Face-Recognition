import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

st.title("Face Recognition System")

model = load_model("model/face_model.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

img_file = st.camera_input("Ambil gambar wajah")

if img_file:
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (128,128)) / 255.0
        face = face.reshape(1,128,128,1)
        pred = model.predict(face)
        st.success(f"Prediction index: {np.argmax(pred)}")

    st.image(img, channels="BGR")
