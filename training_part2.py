import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle

# Paths
DATA_PATH = 'isl_landmarks1'  # Folder containing .npy files for landmarks


# Load the data from .npy files and labels
def load_data(data_path):
    X = []
    y = []
    for file in os.listdir(data_path):
        if file.endswith('.npy'):
            landmarks = np.load(os.path.join(data_path, file))
            X.append(landmarks)  # Load 3D landmarks (2 hands, 21 landmarks, 3D coords)

            # The label is assumed to be part of the filename (e.g., A_image1.npy, B_image2.npy)
            label = file.split('_')[0]
            y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, y


# Load the data and labels
X, y = load_data(DATA_PATH)

# Reshape the input to fit the CNN
X = X.reshape(-1, 2, 21, 3)  # Reshape to (num_samples, 2 hands, 21 landmarks, 3 coordinates)

# Normalize the data to the range [0, 1]
X = X / np.amax(X)  # Normalize by dividing by the max value in the dataset

# Binarize the labels (One-Hot Encoding)
lb = LabelBinarizer()
y = lb.fit_transform(y)  # Convert labels to one-hot encoded format

# Split data into training and testing sets
X, y = shuffle(X, y, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Adjust input shape, and minimize pooling/kernels to avoid shrinking input too much
input_shape = (2, 21, 3)

# Define the CNN model for 3D landmarks
model = models.Sequential()

# Ensure valid input shape
model.add(layers.Input(shape=input_shape))

# Convolutional Layers
model.add(layers.Conv2D(32, (1, 2), activation='relu', padding='same'))  # Use padding='same' to preserve height
model.add(layers.MaxPooling2D((1, 2)))  # Adjust pooling size
model.add(layers.Conv2D(64, (1, 2), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((1, 2)))

# Flatten and Dense Layers
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(len(lb.classes_), activation='softmax'))  # Output layer for N classes (alphabets)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the model for future use
model.save('hand_alphabet_cnn_model_3d_fixed.h5')

# Print out label classes for future reference
print("Label classes:", lb.classes_)
