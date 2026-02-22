import os
import cv2
import numpy as np
import json
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # ...\dezo\src
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # ...\dezo

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")           # ...\dezo\dataset
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")               # ...\dezo\models
os.makedirs(MODEL_DIR, exist_ok=True)

print("[INFO] Dataset path:", DATASET_PATH)
print("[INFO] Model path:", MODEL_DIR)

faces = []
labels = []
label_dict = {}
label_id = 0

if not os.path.exists(DATASET_PATH):
    print("[ERROR] Dataset folder does not exist. Run dataset_creator.py first.")
    exit(1)

print("[INFO] Loading dataset...")

for person_name in os.listdir(DATASET_PATH):
    person_dir = os.path.join(DATASET_PATH, person_name)
    if not os.path.isdir(person_dir):
        continue

    print(f"[INFO] Reading images for person: {person_name}")
    label_dict[label_id] = person_name

    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Could not read image: {img_path}")
            continue

        faces.append(img)
        labels.append(label_id)

    label_id += 1

if len(faces) == 0:
    print("[ERROR] No images found in dataset. Please capture faces first.")
    exit(1)

faces = np.array(faces)
labels = np.array(labels)

print(f"[INFO] Total faces loaded: {len(faces)}")
print(f"[INFO] Unique persons: {len(label_dict)}")
print("[INFO] Training LBPH recognizer...")

recognizer = cv2.face.LBPHFaceRecognizer_create(
    radius=1,
    neighbors=8,
    grid_x=8,
    grid_y=8
)

recognizer.train(faces, labels)

model_file = os.path.join(MODEL_DIR, "face_model.yml")
labels_file = os.path.join(MODEL_DIR, "labels.json")

recognizer.write(model_file)
label_dict_str_keys = {str(k): v for k, v in label_dict.items()}

with open(labels_file, "w") as f:
    json.dump(label_dict_str_keys, f)

print("[INFO] Training complete.")
print(f"[INFO] Saved model to: {model_file}")
print(f"[INFO] Saved labels to: {labels_file}")
