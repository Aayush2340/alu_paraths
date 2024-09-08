import cv2
import mediapipe as mp
import os

# Initialize MediaPipe Hand model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Directory to save landmarks
landmarks_dir = "hand_landmarks"

# Create directory if it doesn't exist
if not os.path.exists(landmarks_dir):
    os.makedirs(landmarks_dir)


def extract_landmarks_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    label_dir = os.path.join(landmarks_dir, label)

    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, _ = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy])

                # Save landmarks as a .txt file
                with open(os.path.join(label_dir, f'{frame_number}.txt'), 'w') as f:
                    for landmark in landmarks:
                        f.write(f"{landmark[0]} {landmark[1]}\n")

                frame_number += 1

    cap.release()


# Example usage for a video file with label "hello"
extract_landmarks_from_video("path_to_your_video.mp4", "hello")
