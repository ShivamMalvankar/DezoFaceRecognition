import os
import cv2
import json
import numpy as np
import time

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # ...\dezo\src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # ...\dezo

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")               # ...\dezo\models
model_file = os.path.join(MODEL_DIR, "face_model.yml")
labels_file = os.path.join(MODEL_DIR, "labels.json")

print("[INFO] Model dir:", MODEL_DIR)

# -----------------------------
# LOAD MODEL + LABELS
# -----------------------------
if not os.path.exists(model_file):
    print("[ERROR] Model file not found. Run train_face_model.py first.")
    exit(1)

if not os.path.exists(labels_file):
    print("[ERROR] Labels file not found. Run train_face_model.py first.")
    exit(1)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(model_file)

with open(labels_file, "r") as f:
    label_dict = json.load(f)  # keys: strings "0", "1", ...

print("[INFO] Loaded model and labels:")
print(label_dict)

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)



# -----------------------------
# CAMERA SETUP
# -----------------------------
cap = cv2.VideoCapture(0)  # change index if needed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("[INFO] Starting real-time identification. Press 'q' to quit.")
fps_time = time.time()

CONFIDENCE_THRESHOLD = 60  # lower is better (tune if needed)

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # -----------------------------
    # FACE DETECTION
    # -----------------------------
    faces_frontal = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    faces_profile = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # Detect right-profile faces using flip
    flipped_gray = cv2.flip(gray, 1)
    faces_profile_flipped = profile_cascade.detectMultiScale(
        flipped_gray,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # Correct flipped coordinates
    corrected_profiles = []
    for (x, y, fw, fh) in faces_profile_flipped:
        corrected_profiles.append((w - x - fw, y, fw, fh))

    # Combine all detections
    faces_detected = []
    faces_detected.extend(faces_frontal)
    faces_detected.extend(faces_profile)
    faces_detected.extend(corrected_profiles)

    # -----------------------------
    # FACE RECOGNITION
    # -----------------------------
    for (x, y, fw, fh) in faces_detected:
        roi_gray = gray[y:y+fh, x:x+fw]

        # Safety check
        if roi_gray.size == 0:
            continue

        roi_gray = cv2.resize(roi_gray, (200, 200))

        label, confidence = recognizer.predict(roi_gray)
        predicted_name = label_dict.get(str(label), "Unknown")

        if confidence < CONFIDENCE_THRESHOLD:
            text = f"{predicted_name} ({int(confidence)})"
            color = (0, 255, 0)

        else:
            text = f"Unknown ({int(confidence)})"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), color, 2)
        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

    # -----------------------------
    # FPS DISPLAY
    # -----------------------------
    fps = 1.0 / (time.time() - fps_time)
    fps_time = time.time()
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2
    )

    cv2.imshow("Real-Time Face Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
print("[INFO] Identification stopped.")
