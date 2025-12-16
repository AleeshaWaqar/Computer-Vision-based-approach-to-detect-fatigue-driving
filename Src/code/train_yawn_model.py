import os
import cv2
import time
import numpy as np
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================
# DATASET PATHS (yours)
# =========================
YAWN_DIR = r"C:\Users\asus\Documents\mouth\yawn"
NO_YAWN_DIR = r"C:\Users\asus\Documents\mouth\no yawn"

# =========================
# TRAINING SETTINGS
# =========================
IMG_SIZE = 64          # keep small for Jetson feasibility
EPOCHS = 12
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

# Output
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # ...\fatigue
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

OUT_MODEL = MODELS_DIR / "yawn_cnn.h5"

def load_images(folder, label):
    X, y = [], []
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = [p for p in folder.rglob("*") if p.suffix.lower() in exts]

    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        X.append(img)
        y.append(label)

    X = np.array(X, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]  # (N, H, W, 1)
    y = np.array(y, dtype=np.int32)
    return X, y

def build_model():
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")  # binary
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def main():
    print("Loading images...")
    X1, y1 = load_images(YAWN_DIR, 1)
    X0, y0 = load_images(NO_YAWN_DIR, 0)

    X = np.concatenate([X1, X0], axis=0)
    y = np.concatenate([y1, y0], axis=0)

    # shuffle
    rng = np.random.default_rng(SEED)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    X, y = X[idx], y[idx]

    print("Total samples:", len(X))
    print("Yawn:", int(y.sum()), "No-yawn:", int((y == 0).sum()))
    print("IMG_SIZE:", IMG_SIZE)

    # split
    n_val = int(len(X) * VAL_SPLIT)
    X_val, y_val = X[:n_val], y[:n_val]
    X_train, y_train = X[n_val:], y[n_val:]

    model = build_model()
    model.summary()

    start = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1
    )
    print("Training time (s):", round(time.time() - start, 2))

    model.save(str(OUT_MODEL))
    print("Saved:", OUT_MODEL)

    # quick eval
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    print("Val accuracy:", round(float(acc), 4))

if __name__ == "__main__":
    main()
