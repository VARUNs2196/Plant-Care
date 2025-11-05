# PlantCare: Real-Time Plant Health and Disease Detection

## 🌿 Project Overview

**PlantCare** is a cross-platform mobile application built with **Flutter** that leverages on-device **TensorFlow Lite (TFLite)** models to provide real-time, instant analysis of plant health. By using the device's camera, users can quickly detect diseases, identify weeds, or determine the edibility of mushrooms, making sophisticated agricultural and botanical knowledge accessible to hobbyists and farmers alike.

This project demonstrates the power of integrating high-performance deep learning models directly into a mobile environment for edge computing.

---

## ✨ Key Features

* **Real-Time Detection:** Utilizes the device camera to perform live inference using TFLite.

* **Multi-Feature Analysis:** Supports three distinct detection modes, each powered by a specialized model:

    * **Disease Prediction (38 Classes):** YOLOv8-based model for identifying common plant diseases (e.g., Apple Scab, Tomato Blight).
    * **Weed Detection (2 Classes):** Specialized model to distinguish between crops and common invasive weeds.
    * **Mushroom Edibility Check (2 Classes):** Determines if a mushroom is likely edible or poisonous.

* **Native Performance:** Built with Flutter for a beautiful, performant UI and utilizes the `tflite_flutter` package for optimized native inference speed.

---

## 💻 Technical Stack

* **Framework:** Flutter (Dart)
* **Mobile ML Library:** `tflite_flutter` (for fast, on-device inference)
* **Model Training Framework:** PyTorch / Ultralytics YOLOv8
* **Core Logic:** `main.dart` (UI, Camera), `tf_lite.dart` (Inference Logic)

---

## 🧠 Model Training & Performance

The detection models deployed in the app were trained using PyTorch with the Ultralytics YOLOv8 framework and aggressively optimized using **INT8 Quantization** for small size and fast performance on mobile devices.

### Training Notebooks

| Feature | Notebook | Input Image Size | Target Classes |
| :--- | :--- | :--- | :--- |
| **Disease Prediction** | [`plantcare-final (1).ipynb`](plantcare-final%20(1).ipynb) | 640x640 | 38 (Apple, Corn, Tomato, etc.) |
| **Weed Detection** | [`weeddetects (1).ipynb`](weeddetects%20(1).ipynb) | 640x640 | 2 (BroWeed, NarWeed) |
| **Mushroom Edibility** | [`notebook98504bb4d5.ipynb`](notebook98504bb4d5.ipynb) | 768x768 | 2 (Edible, Poisonous) |

### Deployed Model Performance Metrics

| Feature | mAP50 | mAP50-95 | Input Size | On-Device Model Size |
| :--- | :--- | :--- | :--- | :--- |
| **Disease Prediction** | **64.7%** | **70.3%** | 640x640 | **~3.2 MB** (`best_int8.tflite`) |
| **Weed Detection** | **61.3%** | **33.7%** | 640x640 | **3.1 MB** (`best_int8.tflite`) |
| **Mushroom Edibility** | **62.3%** | **62.3%** | 768x768 | **3.22 MB** (`best_int8.tflite`) |

---

## 💻 Application Logic Details

### 1. TFLite Implementation (`tf_lite.dart`)
The `TFLiteDetector` handles all ML pipeline steps:
1.  **Loading:** Asynchronously loads the quantized TFLite model from the Flutter assets using `tflite_flutter`.
2.  **Input:** Resizes the camera frame to the model's required input size (e.g., 640x640 or 768x768) and normalizes pixel values (0-255 to 0.0-1.0).
3.  **Post-Processing:** Implements a simplified YOLO output parsing method, iterating through the 8400 anchor predictions to find the bounding box and class score with the highest confidence above a set threshold (currently `0.3`).

### 2. Camera Integration (`main.dart`)
The `FeatureScreen` is responsible for handling the live detection loop. It initializes the camera controller and uses a `Timer.periodic` function to periodically capture an image (`_controller!.takePicture()`), run the TFLite inference on that image's file path, and update the UI with the detection results.

---

## 🚀 Getting Started

### Prerequisites

* Flutter SDK (v3.19.0 or higher)
* A physical device (Android/iOS) or emulator with a functional camera.

### Installation and Setup

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/VARUNs2196/Plant-Care
    cd PlantCare
    ```

2.  **Add TFLite Models:**
    You must create an `assets/models` folder and place the required TFLite files inside it. The application expects the following filenames, which are typical outputs from YOLOv8 INT8 quantization:
    ```
    /PlantCare
    └── assets
        └── models
            ├── best_int8.tflite   # Used for Plant Disease
            ├── best_float16.tflite # Used for Weed Detection
            └── mushroom.tflite    # Used for Mushroom Edibility
    ```
    *(Note: Ensure your `pubspec.yaml` correctly includes the `assets/models` folder).*

3.  **Get Dependencies:**
    ```bash
    flutter pub get
    ```

4.  **Run the App:**
    ```bash
    flutter run
    ```
    *Due to camera requirements, it is recommended to run this on a physical device.*

---

## 📸 Screenshots


### Home Screen
<img width="220" height="476" alt="image" src="https://github.com/user-attachments/assets/d2ddaf7f-47ce-4f20-b3a1-657c1a3452a4" />




### Disease Prediction Feature
<img width="220" height="476" alt="image" src="https://github.com/user-attachments/assets/5ac0ed44-4acd-4104-81b7-d71c224e8686" />




### Weed Detection Feature
<img width="220" height="476" alt="image" src="https://github.com/user-attachments/assets/e684300d-3013-4dde-8764-726ef00770df" />

### Mushroom Edibility Feature
<img width="220" height="476" alt="image" src="https://github.com/user-attachments/assets/4eab2891-52c0-44a0-aa8c-82debc89cd16" />


