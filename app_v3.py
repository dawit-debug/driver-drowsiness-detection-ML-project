import cv2
import numpy as np
import tensorflow as tf
import pygame
import time
from collections import deque

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# -----------------------------
# Initialize pygame
# -----------------------------
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm/alarm.wav")

# -----------------------------
# Load Model (v3 - trained on combined DDD + MRL eye crops, 64x64
# grayscale. Rescaling is built into the model itself, so we do NOT
# divide by 255 before predict.)
# -----------------------------
model = tf.keras.models.load_model("model/eye_model_v3.keras")

EYE_CROP_SIZE = (64, 64)

# -----------------------------
# MediaPipe FaceLandmarker (Tasks API)
# Download once: https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
# and place it next to this script as "face_landmarker.task"
# -----------------------------
base_options = mp_python.BaseOptions(model_asset_path="face_landmarker.task")
landmarker_options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)
face_landmarker = mp_vision.FaceLandmarker.create_from_options(landmarker_options)

# Same eye landmark indices + padding used when building the training data
LEFT_EYE = [33, 160, 158, 133, 153, 144, 163, 7]
RIGHT_EYE = [362, 385, 387, 263, 373, 380, 249, 466]
PADDING_RATIO = 0.35


def get_eye_crop(frame_rgb, landmarks, indices, img_w, img_h):
    pts = [
        (int(landmarks[i].x * img_w), int(landmarks[i].y * img_h))
        for i in indices
    ]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min
    pad_w = int(bw * PADDING_RATIO) + 2
    pad_h = int(bh * PADDING_RATIO) + 2

    x1 = max(0, x_min - pad_w)
    x2 = min(img_w, x_max + pad_w)
    y1 = max(0, y_min - pad_h)
    y2 = min(img_h, y_max + pad_h)

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    return frame_rgb[y1:y2, x1:x2], (x1, y1, x2, y2)


def predict_eye(crop_rgb):
    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, EYE_CROP_SIZE)
    gray = gray.astype("float32")  # model has Rescaling built in - no /255 here
    gray = np.expand_dims(gray, axis=-1)  # add channel dim -> (64,64,1)
    gray = np.expand_dims(gray, axis=0)   # add batch dim -> (1,64,64,1)
    return float(model.predict(gray, verbose=0)[0][0])


# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# -----------------------------
# Variables
# -----------------------------
drowsy_start = None
alarm_playing = False

# Temporal smoothing: require most of the last N frames to agree
# before calling it DROWSY, so a normal blink (1-2 frames) can't
# flip the status. Tune WINDOW_SIZE/DROWSY_RATIO_THRESHOLD to taste.
WINDOW_SIZE = 8
DROWSY_RATIO_THRESHOLD = 0.7
prediction_window = deque(maxlen=WINDOW_SIZE)

prev_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame_rgb.shape

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = face_landmarker.detect(mp_image)

    if result.face_landmarks:

        landmarks = result.face_landmarks[0]

        left_crop, left_box = get_eye_crop(frame_rgb, landmarks, LEFT_EYE, w, h)
        right_crop, right_box = get_eye_crop(frame_rgb, landmarks, RIGHT_EYE, w, h)

        eye_predictions = []
        if left_crop is not None and left_crop.size > 0:
            eye_predictions.append(predict_eye(left_crop))
        if right_crop is not None and right_crop.size > 0:
            eye_predictions.append(predict_eye(right_crop))

        if eye_predictions:

            # Average both eyes when we have both; falls back to
            # whichever single eye is visible (e.g. one blocked by glare).
            prediction = sum(eye_predictions) / len(eye_predictions)

            print("Prediction:", prediction, "eyes used:", len(eye_predictions))

            # class 0 = closed, class 1 = open (alphabetical folder order
            # from training) - so prediction < 0.5 means closed/drowsy.
            is_drowsy_frame = prediction < 0.5
            prediction_window.append(is_drowsy_frame)

            drowsy_ratio = sum(prediction_window) / len(prediction_window)
            smoothed_drowsy = (
                len(prediction_window) == WINDOW_SIZE
                and drowsy_ratio >= DROWSY_RATIO_THRESHOLD
            )

            if smoothed_drowsy:

                status = "DROWSY"
                color = (0, 0, 255)
                confidence = (1 - prediction) * 100

                if drowsy_start is None:
                    drowsy_start = time.time()

                elapsed = time.time() - drowsy_start

                if elapsed >= 2:
                    if not alarm_playing:
                        alarm.play(-1)
                        alarm_playing = True

            else:

                status = "ALERT"
                color = (0, 255, 0)
                confidence = prediction * 100

                drowsy_start = None

                if alarm_playing:
                    alarm.stop()
                    alarm_playing = False

            # Draw boxes around the eyes we used
            for box in (left_box, right_box):
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            cv2.putText(
                frame, f"{status}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
            )
            cv2.putText(
                frame, f"Confidence: {confidence:.2f}%", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

    cv2.putText(
        frame, f"FPS: {int(fps)}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2
    )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if alarm_playing:
    alarm.stop()

cap.release()
cv2.destroyAllWindows()