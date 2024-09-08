import cv2
import mediapipe as mp
import numpy as np
import os

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Paths
DATA_PATH = 'isl_landmarks1'  # Folder to save landmarks
IMAGE_BASE_PATH = 'nwe_data'  # Base folder containing folders for each alphabet (A, B, C, etc.)

# Create directory for storing landmarks if it doesn't exist
os.makedirs(DATA_PATH, exist_ok=True)

# Check if IMAGE_BASE_PATH exists, if not, create it
if not os.path.exists(IMAGE_BASE_PATH):
    os.makedirs(IMAGE_BASE_PATH)
    print(f"Directory '{IMAGE_BASE_PATH}' not found. Created new directory.")

# Function to extract hand landmarks in 3D from an image
def extract_landmarks_from_image(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_image)

    landmarks_list = []  # To store landmarks for both hands
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                h, w, _ = image.shape
                # Get x, y, z (z is relative depth)
                cx, cy, cz = int(lm.x * w), int(lm.y * h), lm.z
                landmarks.append((cx, cy, cz))  # Store 3D coordinates
            landmarks_list.append(landmarks)

    # Ensure both hands are represented (add zero padding if only one hand is detected)
    while len(landmarks_list) < 2:
        landmarks_list.append([(0, 0, 0)] * 21)  # Zero padding for the second hand

    return np.array(landmarks_list)

# Process each image in the alphabet folders
for label in os.listdir(IMAGE_BASE_PATH):
    alphabet_folder = os.path.join(IMAGE_BASE_PATH, label)

    if os.path.isdir(alphabet_folder):  # Ensure it's a directory
        for image_file in os.listdir(alphabet_folder):
            if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):  # Handle case-insensitive extensions
                image_path = os.path.join(alphabet_folder, image_file)
                landmarks = extract_landmarks_from_image(image_path)

                # Save landmarks to a file with the label (alphabet name)
                np.save(os.path.join(DATA_PATH, f'{label}_{image_file}.npy'), landmarks)
                print(f'Extracted and saved landmarks for {image_file} from alphabet {label}')

# Release resources
hands.close()
