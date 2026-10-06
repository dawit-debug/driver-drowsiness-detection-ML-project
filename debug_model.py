"""
Sanity check: run the saved model on a few known images from your
training dataset folders (NOT the webcam) to see if the model itself
gives sensible predictions.

Edit DATASET_PATH below to point at the folder that contains your
"Drowsy" and "Non Drowsy" subfolders (the same one you used in Colab).
"""

import os
import random
import cv2
import numpy as np
import tensorflow as tf

DATASET_PATH = "C:\\Users\\Administrator\\Downloads\\Driver Drowsiness Dataset (DDD)"  # <-- edit this

MODEL_PATH = "model/drowsiness_model_v2.keras"

model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128, 128))
    
    img = np.expand_dims(img, axis=0)
    pred = float(model.predict(img, verbose=0)[0][0])
    return pred

for label in ["Drowsy", "Non Drowsy"]:
    folder = os.path.join(DATASET_PATH, label)
    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        continue

    samples = random.sample(os.listdir(folder), min(5, len(os.listdir(folder))))

    print(f"\n--- {label} samples (expect close to {'0' if label == 'Drowsy' else '1'}) ---")
    for s in samples:
        p = predict_image(os.path.join(folder, s))
        print(f"{s}: {p:.4f}")