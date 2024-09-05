import face_recognition
import cv2
import numpy as np
import os
import threading
from queue import Queue


# Load known face encodings and names from the images directory
def load_known_faces(directory='images'):
    known_face_encodings = []
    known_face_names = []
    for img in os.listdir(directory):
        image = face_recognition.load_image_file(os.path.join(directory, img))
        encodings = face_recognition.face_encodings(image)
        if encodings:  # Ensure there's at least one encoding
            known_face_encodings.append(encodings[0])
            known_face_names.append(img)
    return known_face_encodings, known_face_names


known_face_encodings, known_face_names = load_known_faces()

video_capture = cv2.VideoCapture(0)

face_locations = []
face_encodings = []
face_names = []
frame_queue = Queue(maxsize=5)  # Limit the number of frames to avoid excessive memory usage


def process_frames():
    global face_locations, face_encodings, face_names
    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Detect face locations and encodings using the HOG model (fastest option)
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Compute distances between detected face and known faces
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Assign name based on best match
            if face_distances[best_match_index] < 0.6:  # A threshold to ensure accuracy
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

        frame_queue.task_done()


# Start the processing thread
processing_thread = threading.Thread(target=process_frames)
processing_thread.daemon = True  # Allow the thread to be killed when the main program exits
processing_thread.start()

while True:
    ret, frame = video_capture.read()

    if not frame_queue.full():
        frame_queue.put(frame)  # Add the captured frame to the queue

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the processing thread
frame_queue.put(None)
processing_thread.join()

video_capture.release()
cv2.destroyAllWindows()
