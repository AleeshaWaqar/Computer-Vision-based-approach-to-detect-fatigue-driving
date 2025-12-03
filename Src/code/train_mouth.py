import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

IMG_SIZE = 48

def load_images(folder, label):
    data = []
    for img in os.listdir(folder):
        path = os.path.join(folder, img)
        im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (IMG_SIZE, IMG_SIZE))
        data.append((im, label))
    return data

print("TRAINING SCRIPT STARTED")
print("Loading dataset...")

open_mouth = load_images(r"C:\Users\asus\Desktop\fatigue\mouth_dataset_medium\mouth_open", 1)
closed_mouth = load_images(r"C:\Users\asus\Desktop\fatigue\mouth_dataset_medium\mouth_closed", 0)

data = open_mouth + closed_mouth
np.random.shuffle(data)

X = np.array([i[0] for i in data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array([i[1] for i in data])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print("Building model...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("Training model...")
model.fit(X_train, y_train, epochs=10, validation_split=0.1)

print("Evaluating model...")
loss, acc = model.evaluate(X_test, y_test)
print("Accuracy:", acc)

print("Saving model...")
model.save(r"C:\Users\asus\Desktop\fatigue\models\mouth_model.h5")
print("Model saved at ../models/mouth_model.h5")
