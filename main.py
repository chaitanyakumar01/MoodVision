import cv2
cv2.setNumThreads(0)

import numpy as np
from deepface import DeepFace
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import threading

# ... (Baaki code same rahega) ...
# -----------------------------------------------------------
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(page_title="MoodVision AI", page_icon="🎭", layout="wide")

st.title("🎭 MoodVision: Real-Time Neural Sentiment Analysis")
st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1 { color: #00FFCC; }
    .stButton>button { width: 100%; border-radius: 20px; }
</style>
""", unsafe_allow_html=True)

st.write("Wait for the camera to load, then click 'Start'. Allow browser permissions if asked.")

# -----------------------------------------------------------
# 2. EMOTION SETTINGS
# -----------------------------------------------------------
emotion_style = {
    'happy':    {'color': (0, 255, 255), 'emoji': 'Happy :)'},
    'sad':      {'color': (255, 0, 0),   'emoji': 'Sad :('},
    'angry':    {'color': (0, 0, 255),   'emoji': 'Angry >:@'},
    'surprise': {'color': (0, 255, 0),   'emoji': 'Wow :O'},
    'neutral':  {'color': (200, 200, 200), 'emoji': 'Normal :|'},
    'fear':     {'color': (128, 0, 128), 'emoji': 'Scared o_O'},
    'disgust':  {'color': (0, 128, 255), 'emoji': 'Yuck XP'}
}

# -----------------------------------------------------------
# 3. VIDEO PROCESSOR CLASS (The Brain)
# -----------------------------------------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # Convert frame to numpy array (OpenCV format)
        img = frame.to_ndarray(format="bgr24")
        
        # Mirror effect
        img = cv2.flip(img, 1)

        try:
            # DeepFace logic (Same as your local code)
            # enforce_detection=False prevents crashes when no face is found
            analysis = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)

            for face in analysis:
                x = face['region']['x']
                y = face['region']['y']
                w = face['region']['w']
                h = face['region']['h']
                
                current_mood = face['dominant_emotion']
                
                if current_mood in emotion_style:
                    style = emotion_style[current_mood]
                    box_color = style['color']
                    display_text = style['emoji']
                else:
                    box_color = (255, 255, 255)
                    display_text = current_mood

                # Drawing
                cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 2)
                cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                cv2.putText(img, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

        except Exception as e:
            # Pass silently to avoid cluttering logs
            pass

        # Return the processed frame back to the browser
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -----------------------------------------------------------
# 4. WEBRTC STREAMER (The Camera Component)
# -----------------------------------------------------------
webrtc_streamer(
    key="moodvision",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("---")
st.caption("Developed with ❤️ by Chaitanya Kumar | Powered by DeepFace & Streamlit")