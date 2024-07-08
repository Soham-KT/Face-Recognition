import face_recognition
import cv2
import numpy as np
import os

images_list = os.listdir('images')

known_face_encodings = []
known_face_names = []

for img in images_list:
    image = face_recognition.load_image_file(os.path.join('images', img))
    known_face_encodings.append(face_recognition.face_encodings(image)[0])
    known_face_names.append(img)
