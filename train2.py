import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

# Load the trained model
model = tf.keras.models.load_model('isl_model.h5')

# Load the encoder to decode the predicted labels
encoder = LabelBinarizer()
encoder.fit(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))  # Assuming you're working with the 26 alphabets

# Function to predict the alphabet based on the extracted landmarks
def predict_alphabet(landmarks):
    # Ensure landmarks contain both hands (2 hands * 21 landmarks * 2 coordinates)
    if landmarks.shape != (2, 21, 2):  # Expecting landmarks for 2 hands
        # Pad with zeros if only one hand is detected or shape is incorrect
        padded_landmarks = np.zeros((2, 21, 2))
        padded_landmarks[:landmarks.shape[0], :, :] = landmarks  # Fill in available landmarks
        landmarks = padded_landmarks

    # Flatten the landmarks and ensure the input has shape (1, 84)
    landmarks_flattened = landmarks.flatten().reshape(1, 84)

    # Predict the alphabet
    prediction = model.predict(landmarks_flattened)
    predicted_label = encoder.inverse_transform(prediction)[0]
    return predicted_label

# Example: Load a landmark file for testing
landmarks = np.load('isl_landmarks/A_test_image.npy')  # Load a test landmark file

# Predict the alphabet
predicted_alphabet = predict_alphabet(landmarks)
print(f"Predicted Alphabet: {predicted_alphabet}")
