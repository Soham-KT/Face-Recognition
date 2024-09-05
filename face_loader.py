import face_recognition
import os


def load_known_faces(directory='images'):
    known_face_encodings = []
    known_face_names = []

    for img in os.listdir(directory):
        image = face_recognition.load_image_file(os.path.join(directory, img))
        encodings = face_recognition.face_encodings(image)

        if encodings:
            known_face_encodings.append(encodings[0])
            known_face_names.append(img)

    return known_face_encodings, known_face_names
