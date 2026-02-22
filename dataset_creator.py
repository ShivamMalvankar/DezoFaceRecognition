# ===============================================
# DATASET CREATION SCRIPT (MULTI-ANGLE)
# Location: dezo/src/dataset_creator.py
# ===============================================

import cv2
import os

# -----------------------------
# PATH SETUP
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../dezo/src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../dezo
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")           # .../dezo/dataset

os.makedirs(DATASET_PATH, exist_ok=True)

print("[INFO] Dataset root:", DATASET_PATH)

# -----------------------------
# INPUT PERSON NAME
# -----------------------------
person_name = input("Enter the name of the person: ").strip()
person_folder = os.path.join(DATASET_PATH, person_name)
os.makedirs(person_folder, exist_ok=True)

print(f"[INFO] Saving images to: {person_folder}")

# -----------------------------
# LOAD CASCADES
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)

if face_cascade.empty():
    print("[ERROR] Failed to load frontal face cascade.")
if profile_cascade.empty():
    print("[ERROR] Failed to load profile face cascade.")

# -----------------------------
# INITIALIZE CAMERA
# -----------------------------
cap = cv2.VideoCapture(0)  # change index if needed
if not cap.isOpened():
    print("[ERROR] Could not open camera.")
    exit(1)

count = 0
MAX_IMAGES = 100

print(f"[INFO] Capturing faces for {person_name}.")
print("[INFO] Press 'q' to quit manually.")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from camera.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # ----- DETECT FACES (FRONTAL) -----
    faces_frontal = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(60, 60)
    )

    # ----- DETECT FACES (PROFILE - LEFT) -----
    faces_profile_left = profile_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # ----- DETECT FACES (PROFILE - RIGHT via FLIP) -----
    flipped = cv2.flip(gray, 1)
    faces_profile_right_raw = profile_cascade.detectMultiScale(
        flipped,
        scaleFactor=1.2,
        minNeighbors=4,
        minSize=(60, 60)
    )

    # Map flipped coords back to original
    faces_profile_right = []
    for (x, y, fw, fh) in faces_profile_right_raw:
        corrected_x = w - x - fw
        faces_profile_right.append((corrected_x, y, fw, fh))

    # Combine all detections
    faces_detected = []
    faces_detected.extend(faces_frontal)
    faces_detected.extend(faces_profile_left)
    faces_detected.extend(faces_profile_right)

    # -----------------------------
    # SAVE FACE IMAGES
    # -----------------------------
    for (x, y, fw, fh) in faces_detected:
        face_roi = gray[y:y+fh, x:x+fw]
        if face_roi.size == 0:
            continue

        face_resized = cv2.resize(face_roi, (200, 200))

        count += 1
        file_path = os.path.join(person_folder, f"{count}.jpg")
        cv2.imwrite(file_path, face_resized)

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Image {count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        print(f"[INFO] Captured image {count}: {file_path}")

        if count >= MAX_IMAGES:
            break

    # -----------------------------
    # SHOW WINDOW
    # -----------------------------
    cv2.imshow("Dataset Creator", frame)

    # Exit if 'q' pressed or max images reached
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= MAX_IMAGES:
        break

# -----------------------------
# CLEANUP
# -----------------------------
cap.release()
cv2.destroyAllWindows()
print(f"[INFO] Finished capturing. Total images: {count}")
print(f"[INFO] Dataset created for {person_name} in {person_folder}")
