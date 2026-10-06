import cv2
import numpy as np
import tensorflow as tf
import pygame
import time
from collections import deque

# -----------------------------
# Initialize pygame
# -----------------------------
pygame.mixer.init()
alarm = pygame.mixer.Sound("alarm/alarm.wav")

# -----------------------------
# Load Model (v2 - trained with augmentation; rescaling is built
# into the model itself, so we do NOT divide by 255 before predict)
# -----------------------------
model = tf.keras.models.load_model("model/drowsiness_model_v2.keras")

# -----------------------------
# Face Detector
# -----------------------------
face_detector = cv2.CascadeClassifier(
    "haarcascade/haarcascade_frontalface_default.xml"
)

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100,100)
    )

    for (x, y, w, h) in faces:

        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (128,128))
        cv2.imwrite("face_to_cnn.jpg", face)
        cv2.imshow("Face Sent to CNN", face)

        # Training used image_dataset_from_directory, which decodes images
        # as RGB. OpenCV frames are BGR, so we must convert before feeding
        # the model or the color channels will be scrambled relative to
        # what the model learned on.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        # v2 model has Rescaling(1./255) built in - do NOT divide here,
        # or the input gets rescaled twice and predictions break.
        face = face.astype("float32")
        face = np.expand_dims(face, axis=0)

        prediction = float(model.predict(face, verbose=0)[0][0])

        print("Prediction:", prediction)

        # --------------------------------------------------
        # IMPORTANT:
        # If labels are reversed, swap these two blocks.
        # --------------------------------------------------

        # Raw per-frame call (used only for smoothing input, not directly
        # for status/alarm decisions - a single blink shouldn't count).
        is_drowsy_frame = prediction < 0.5
        prediction_window.append(is_drowsy_frame)

        drowsy_ratio = sum(prediction_window) / len(prediction_window)
        smoothed_drowsy = (
            len(prediction_window) == WINDOW_SIZE
            and drowsy_ratio >= DROWSY_RATIO_THRESHOLD
        )

        if smoothed_drowsy:

            status = "DROWSY"
            color = (0,0,255)

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
            color = (0,255,0)

            confidence = prediction * 100

            drowsy_start = None

            if alarm_playing:

                alarm.stop()
                alarm_playing = False

        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

        cv2.putText(
            frame,
            f"{status}",
            (x,y-40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"Confidence: {confidence:.2f}%",
            (x,y-15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (20,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255,255,0),
        2
    )

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if alarm_playing:
    alarm.stop()

cap.release()
cv2.destroyAllWindows()