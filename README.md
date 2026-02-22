---

# 🧠 Real-Time Face Recognition System (Multi-Angle)

A complete **face recognition pipeline** built using **OpenCV** that supports:

* 📸 Dataset creation (multi-angle: frontal + profile)
* 🧠 Model training using **LBPH (Local Binary Pattern Histogram)**
* 🎥 Real-time face identification via webcam

---

## 📂 Project Structure

```
dezo/
│
├── dataset/                  # Stores captured face images
│   └── person_name/
│       ├── 1.jpg
│       ├── 2.jpg
│       └── ...
│
├── models/                   # Stores trained model + labels
│   ├── face_model.yml
│   └── labels.json
│
├── src/
│   ├── dataset_creator.py    # Capture dataset
│   ├── train_face_model.py   # Train model
│   └── realtime_face_identification.py  # Live recognition
│
└── requirements.txt
```

---

## ⚙️ Technologies Used

* **Python 3.x**
* **OpenCV (cv2)**
* **NumPy**
* **JSON**
* **Haar Cascade Classifiers**
* **LBPH Face Recognizer**

---

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

If `cv2.face` is missing:

```bash
pip install opencv-contrib-python
```

---

## 📸 Step 1: Dataset Creation

Run:

```bash
python src/dataset_creator.py
```

### 🔹 Features:

* Captures **100 face images**
* Supports:

  * Frontal faces
  * Left profile
  * Right profile (via image flip)
* Automatically:

  * Converts to grayscale
  * Resizes to **200x200**
  * Saves in structured folder

### 🔹 Output:

```
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

### 🔹 What happens:

* Loads dataset images
* Assigns numeric labels to each person
* Trains **LBPH Face Recognizer**

### 🔹 Output:

```
models/
├── face_model.yml
└── labels.json
```

### 🔹 Example labels.json:

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

### 🔹 Features:

* Detects faces using:

  * Frontal Haar Cascade
  * Profile Haar Cascade
* Recognizes faces using trained model
* Displays:

  * Name
  * Confidence score
  * FPS

---

## 🧠 How It Works

### 1. Face Detection

Uses Haar Cascades:

* `haarcascade_frontalface_default.xml`
* `haarcascade_profileface.xml`

### 2. Multi-Angle Detection

* Left profile → direct detection
* Right profile → image flipped → detection → coordinates corrected

### 3. Face Recognition

Uses:
👉 **LBPH (Local Binary Pattern Histogram)**

* Robust to lighting changes
* Works well for real-time systems
* Outputs:

  * `label`
  * `confidence` (lower = better)

---

## ⚖️ Confidence Threshold

```python
CONFIDENCE_THRESHOLD = 60
```

* **< 60 → Recognized**
* **> 60 → Unknown**

👉 You can tune this value for better accuracy.

---

## 🎯 Key Features

✅ Multi-angle face detection
✅ Automatic dataset organization
✅ Real-time recognition
✅ FPS counter
✅ Modular structure
✅ Easy to extend

---

## ⚠️ Common Errors & Fixes

### ❌ `cv2.face not found`

✔ Install:

```bash
pip install opencv-contrib-python
```

---

### ❌ Camera not opening

✔ Try:

```python
cv2.VideoCapture(1)
```

---

### ❌ Model file not found

✔ Run training first:

```bash
python src/train_face_model.py
```

---

### ❌ No faces detected

✔ Improve:

* Lighting conditions
* Camera quality
* Face angle

---

## 🚀 Future Improvements

* 🔐 Face mask detection
* 📊 Attendance system integration
* ☁️ Cloud database (Firebase)
* 📱 Mobile app integration
* 🤖 Deep learning (CNN / FaceNet)

---

## 👨‍💻 Author

**Shivam Malvankar**

---

## ⭐ Tips for Best Accuracy

* Capture dataset in **different lighting conditions**
* Include:

  * Front
  * Left
  * Right
* Avoid blurry images
* Keep face centered

---
