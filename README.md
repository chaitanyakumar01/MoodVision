# MOODVISION: Real-Time Affective Computing Interface

**MoodVision** is a computer vision system designed for real-time facial emotion recognition (FER). Leveraging deep learning algorithms, the system processes live video streams to detect, classify, and visualize human emotional states with low-latency analytics.

## ⚙️ System Architecture

The application operates on a pipeline integrating video capture, frame processing, and neural network inference:

* **Input Layer:** Captures raw video frames via WebRTC.
* **Processing Layer:** Utilizes **DeepFace** (backed by TensorFlow/Keras) for face detection and emotion classification.
* **Visualization Layer:** Renders real-time bounding boxes and updates a dynamic Streamlit dashboard with probability distributions.

## 📋 Technical Specifications

### Key Modules
* **Real-Time Inference:** Frame-by-frame analysis with optimized latency for live feedback.
* **Neural Scans Counter:** Tracks total analyzed frames (Session Data Points).
* **Dynamic Analytics:** Visualizes confidence scores across 7 distinct emotional classes (Happy, Sad, Neutral, Angry, Fear, Surprise, Disgust).
* **GUI:** Custom CSS implementation featuring glassmorphism design language for data readability.

### Tech Stack
| Component | Technology |
| :--- | :--- |
| **Runtime** | Python 3.9+ |
| **Frontend** | Streamlit |
| **Computer Vision** | OpenCV (`cv2`) |
| **Deep Learning** | DeepFace |
| **Streaming** | `streamlit-webrtc` |

## 🚀 Installation & Deployment

### Prerequisites
Ensure `Python` and `pip` are installed in the local environment.

### Local Execution
1.  **Clone Repository**
    ```bash
    git clone [https://github.com/chaitanyakumar01/MoodVision.git](https://github.com/chaitanyakumar01/MoodVision.git)
    cd MoodVision
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Initialize System**
    ```bash
    streamlit run main.py
    ```

## ⚠️ Operational Notes
* **Lighting:** Optimal detection accuracy requires adequate frontal lighting.
* **Latency:** Inference speed depends on local CPU/GPU capabilities.
* **Privacy:** Video data is processed locally in RAM and is not stored persistently.

---
*Maintained by [Chaitanya Kumar](https://github.com/chaitanyakumar01)*
