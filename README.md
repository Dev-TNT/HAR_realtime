# Real-time Human Activity Recognition (HAR) using Mediapipe & Bidirectional GRU

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Latest-green.svg)](https://mediapipe.dev/)

A high-performance, real-time Human Activity Recognition system based on body pose estimation. This project leverages **MediaPipe** for robust landmark extraction and a **Bidirectional GRU** deep learning architecture to classify human actions from live video streams.

---

## 📌 Key Features
* **Real-time Processing:** Optimized pipeline achieving 30+ FPS for seamless inference.
* **Geometric Feature Engineering:** Beyond raw coordinates, the system calculates joint angles and relative distances to improve recognition for nuanced actions (e.g., drinking, clapping).
* **Robust Normalization:** Implements geometric scaling and centering, ensuring the model remains invariant to the user's distance or position relative to the camera.
* **Modern UI/UX:** A sleek dashboard built with `customtkinter`, featuring real-time probability charts and performance metrics.

## 🛠 Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** TensorFlow/Keras (Bidirectional GRU)
* **GUI Framework:** CustomTkinter
* **Data Science:** NumPy, Pandas, Scikit-learn, Matplotlib

## 📂 Project Structure
* `GUI&inference.py`: Main application integrating the UI and the inference engine.
* `make_normalize_data.py`: Data collection utility with automated landmark normalization and feature extraction.
* `Train_GRU_Upgrade.py`: Comprehensive training script featuring Early Stopping, LR reduction, and model evaluation.
* `label_map.json`: Configuration file for action class mapping.

## 🚀 Getting Started

### 1. Installation
```bash
# Clone the repository
git clone [https://github.com/Dev-TNT/HAR_realtime.git](https://github.com/Dev-TNT/HAR_realtime.git)
cd HAR_realtime

# Install required dependencies
pip install opencv-python mediapipe tensorflow customtkinter pandas numpy pillow scikit-learn seaborn matplotlib
