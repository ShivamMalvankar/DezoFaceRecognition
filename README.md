# 🧠 Real-Time Face Recognition System (Multi-Angle)

A complete **face recognition pipeline** built using **OpenCV** that supports:

* 📸 Dataset creation (multi-angle: frontal + profile)
* 🧠 Model training using LBPH (Local Binary Pattern Histogram)
* 🎥 Real-time face identification via webcam

---

## 📂 Project Structure

```text
dezo/
├── dataset/
│   └── person_name/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── models/
│   ├── face_model.yml
│   └── labels.json
│
├── src/
│   ├── dataset_creator.py
│   ├── train_face_model.py
│   └── realtime_face_identification.py
│
└── requirements.txt
```

---

## ⚙️ Technologies Used

* Python 3.x
* OpenCV (`cv2`)
* NumPy
* JSON
* Haar Cascade Classifiers
* LBPH Face Recognizer

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

If you face issues with `cv2.face`, install:

```bash
pip install opencv-contrib-python
```

---

## 📸 Step 1: Create Dataset

Run:

```bash
python src/dataset_creator.py
```

### Features:

* Captures up to **100 images per person**
* Detects:

  * Frontal faces
  * Left profile
  * Right profile (via image flip)
* Automatically:

  * Converts to grayscale
  * Resizes to 200x200
  * Saves in structured folders

### Output Example:

```text
dataset/
└── Shivam/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

---

## 🧠 Step 2: Train Model

Run:

```bash
python src/train_face_model.py
```

### What it does:

* Loads dataset images
* Assigns numeric labels to each person
* Trains LBPH face recognizer

### Output:

```text
models/
├── face_model.yml
└── labels.json
```

### Example `labels.json`:

```json
{
  "0": "Shivam",
  "1": "Rahul"
}
```

---

## 🎥 Step 3: Real-Time Face Identification

Run:

```bash
python src/realtime_face_identification.py
```

### Features:

* Detects faces using:

  * Frontal Haar Cascade
  * Profile Haar Cascade
* Recognizes faces using trained model
* Displays:

  * Name
  * Confidence score
  * FPS (frames per second)

---

## 🧠 How It Works

### Face Detection

Uses Haar Cascade XML files:

* `haarcascade_frontalface_default.xml`
* `haarcascade_profileface.xml`

### Multi-Angle Detection

* Left profile → detected directly
* Right profile → image flipped → detected → coordinates corrected

### Face Recognition

Uses **LBPH (Local Binary Pattern Histogram)**:

* Works well in real-time
* Handles lighting variations
* Returns:

  * Label (person)
  * Confidence (lower = better match)

---

## ⚖️ Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 60
```

* Less than 60 → Recognized
* Greater than 60 → Unknown

You can tune this value based on accuracy needs.

---

## 🎯 Key Features

* Multi-angle face detection
* Automatic dataset organization
* Real-time recognition
* FPS counter
* Clean modular structure
* Easy to extend

---

## ⚠️ Common Issues & Fixes

### `cv2.face` not found

```bash
pip install opencv-contrib-python
```

---

### Camera not opening

Try changing camera index:

```python
cv2.VideoCapture(1)
```

---

### Model file not found

Make sure you run training first:

```bash
python src/train_face_model.py
```

---

### No faces detected

* Improve lighting
* Face camera properly
* Avoid blur

---

## 🚀 Future Improvements

* Face mask detection
* Attendance system integration
* Cloud database (Firebase)
* Mobile app integration
* Deep learning models (FaceNet, CNN)

---

## 👨‍💻 Author

**Shivam Malvankar**

---

## ⭐ Tips for Best Accuracy

* Capture images in different lighting conditions
* Include front, left, and right angles
* Keep face clearly visible
* Avoid motion blur

---
