import face_recognition
import numpy as np
from queue import Queue


def process_frames(frame_queue, known_face_encodings, known_face_names):
    """Processes frames for face detection and recognition."""
    face_locations = []
    face_names = []

    while True:
        frame = frame_queue.get()
        if frame is None:
            break

        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = np.ascontiguousarray(small_frame[:, :, ::-1])

        # Detect face locations and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            # Assign name based on best match with a threshold
            if face_distances[best_match_index] < 0.6:
                name = known_face_names[best_match_index]
            else:
                name = "Unknown"

            face_names.append(name)

        frame_queue.task_done()

    return face_locations, face_names
