import cv2
import threading
from queue import Queue

from face_loader import load_known_faces
from frame_processor import process_frames
from video_display import display_results


def main():
    # Load known faces
    known_face_encodings, known_face_names = load_known_faces()

    # Initialize video capture
    video_capture = cv2.VideoCapture(0)

    frame_queue = Queue(maxsize=5)

    # Start processing thread
    processing_thread = threading.Thread(target=process_frames,
                                         args=(frame_queue, known_face_encodings, known_face_names))
    processing_thread.daemon = True
    processing_thread.start()

    while True:
        ret, frame = video_capture.read()
        if not frame_queue.full():
            frame_queue.put(frame)

        # Process frames and display results
        face_locations, face_names = process_frames(frame_queue, known_face_encodings, known_face_names)
        display_results(frame, face_locations, face_names)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Stop the processing thread
    frame_queue.put(None)
    processing_thread.join()

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
