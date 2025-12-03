import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Correct dataset paths
YAWN_DIR = r"C:\Users\asus\Documents\mouth\yawn"
NOYAWN_DIR = r"C:\Users\asus\Documents\mouth\no yawn"

MODEL_PATH = r"C:\Users\asus\Desktop\fatigue\models\mouth_model.h5"

IMG_SIZE = 24  # <<< IMPORTANT: 24x24 GRAYSCALE

def load(folder, label):
    X, y = [], []
    for file in os.listdir(folder):
        path = os.path.join(folder, file)
        img = cv2.imread(path)
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.reshape(IMG_SIZE, IMG_SIZE, 1)

        X.append(img)
        y.append(label)
    return X, y

print("Loading images...")

X1, y1 = load(YAWN_DIR, 1)
X0, y0 = load(NOYAWN_DIR, 0)

X = np.array(X1 + X0, dtype="float32") / 255.0
y = np.array(y1 + y0)

print("Total images:", X.shape[0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=True
)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

print("Training...")
model.fit(X_train, y_train, epochs=8, validation_split=0.2, batch_size=32)

print("Evaluating...")
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test accuracy: {acc:.4f}")

print("Saving...")
model.save(MODEL_PATH)
print("Saved new mouth model at:", MODEL_PATH)
