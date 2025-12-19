import streamlit as st
import cv2
import numpy as np
import threading
import time
import pandas as pd
import plotly.express as px
import av
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer

# Basic page setup
st.set_page_config(
    page_title="MoodVision", 
    layout="wide", 
    initial_sidebar_state="expanded"
)


# Global variables for styling
BG_GRADIENT = "linear-gradient(135deg, #0f0c29, #302b63, #24243e)"
TEXT_COLOR = "white"
ACCENT_COLOR = "#00C9FF"

@st.cache_resource
def get_shared_state():
    return {
        "counts": {"happy": 0, "sad": 0, "angry": 0, "neutral": 0, "surprise": 0, "fear": 0, "disgust": 0},
        "lock": threading.Lock()
    }

state = get_shared_state()

# Custom CSS 
st.markdown(f"""
<style>
    [data-testid="stHeader"] {{
        background-color: rgba(0,0,0,0); /* Transparent Background */
    }}

    [data-testid="stToolbar"] {{
        visibility: hidden; /* Deploy button gayab */
        right: 2rem;
    }}
    
    [data-testid="stDecoration"] {{
        visibility: hidden;
    }}

    footer {{visibility: hidden;}}
    
    .block-container {{
        padding-top: 2rem !important; 
        padding-bottom: 2rem !important;
    }}

    /* Main background setup */
    .stApp {{
        background: {BG_GRADIENT};
        color: {TEXT_COLOR};
    }}

    

    /* Glass card effect */
    div[data-testid="metric-container"], 
    div[data-testid="stArrowVegaLiteChart"], 
    .glass-card {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: all 0.3s ease;
    }}
    



    /* Hover animation */
    div[data-testid="metric-container"]:hover {{
        transform: translateY(-5px);
        border-color: {ACCENT_COLOR};
    }}

    /* Sidebar tweaks */
    section[data-testid="stSidebar"] {{
        background-color: #0b0f19 !important;
        border-right: 1px solid #334155;
    }}
    



    /* Custom Navbar */
    .navbar {{
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 30px;
        margin-bottom: 25px;
        display: flex;
        align-items: center;
        justify-content: center; 
        border-radius: 20px;
    }}
    

    div.stButton > button {{
        background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
        color: black;
        border: none;
        border-radius: 50px;
        padding: 10px 25px;
        font-weight: 700;
    }}
</style>
""", unsafe_allow_html=True)





# Sidebar layout
with st.sidebar:

    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712035.png", width=50)
    st.title("Control Hub")
    st.markdown("---")
    
    st.markdown("### ⚡ Controls")

    auto_update = st.checkbox("🟢 Live Sync", value=False)
    

    if st.button("🔄 Reset Data"):
        with state["lock"]:

            for key in state["counts"]:
                state["counts"][key] = 0
        st.rerun()
        
    st.markdown("---")
    st.caption("System Status: Online")





st.markdown("""
<div class="navbar">
    <div style="display: flex; align-items: center; gap: 15px;">
        <h1 style="margin:0; font-size: 28px; font-weight: 800; letter-spacing: -0.5px; 
                   background: -webkit-linear-gradient(left, #38BDF8, #A855F7);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;">
            MoodVision <span style="font-weight: 300; opacity: 0.9;"></span>
        </h1>
    </div>
</div>
""", unsafe_allow_html=True)






def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    
    emotion_colors = {
        "happy": (0, 255, 0),       # Green
        "sad": (255, 0, 0),         # Blue
        "angry": (0, 0, 255),       # Red
        "neutral": (0, 255, 255),   # Yellow
        "surprise": (255, 165, 0),  # Sky Blue/Cyan type
        "fear": (128, 0, 128),      # Dark Purple
        "disgust": (0, 128, 0)      # Dark Green
    }
    
    try:
        results = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        
        if results:
            face_data = results[0]
            emotion = face_data['dominant_emotion']
            
            x = face_data['region']['x']
            y = face_data['region']['y']
            w = face_data['region']['w']
            h = face_data['region']['h']

            with state["lock"]:
                if emotion in state["counts"]:
                    state["counts"][emotion] += 1

            box_color = emotion_colors.get(emotion.lower(), (255, 0, 255))

            cv2.rectangle(img, (x, y), (x + w, y + h), box_color, 3)
            
            cv2.putText(img, f"{emotion.upper()}", (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
            
    except Exception as e:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")





# Calculating stats for the top cards
with state["lock"]:
    total_frames = sum(state["counts"].values())
    if total_frames > 0:
        dominant = max(state["counts"], key=state["counts"].get).upper()
    else:
        dominant = "WAITING..."




# Displaying KPIs
kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Neural Scans ", total_frames) 
kpi2.metric("Dominant Mood", dominant)
vibe_dict = {
    "HAPPY": "Radiant 🌟", 
    "SAD": "Blue 🌧️", 
    "ANGRY": "Heated 🔥", 
    "NEUTRAL": "Chill 🍃", 
    "SURPRISE": "Shocked ⚡", 
    "FEAR": "Nervous 😰", 
    "DISGUST": "Eww 🤢",
    "WAITING...": "Loading..."
}

# Current vibe fetch 
current_vibe = vibe_dict.get(dominant, "Unknown")
kpi3.metric("Current Vibe", current_vibe)
st.write("") # Spacer



# Main Content Grid
col1, col2 = st.columns([2, 1])



## Left Side: Camera
with col1:
    st.markdown("### 📷 Live Feed")
    
   
    
    st.markdown("""
        <div style="background: rgba(255, 255, 255, 0.05); 
                    border-radius: 50px; 
                    padding: 5px 20px; 
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    margin-bottom: 15px;
                    text-align: center;">
            <p style="color: #00C9FF; 
                      font-size: 16px; 
                      font-weight: 600; 
                      margin: 0; 
                      letter-spacing: 1px;
                      text-shadow: 0 0 10px rgba(0, 201, 255, 0.3);">
                Show Your Pretty Face Here 👇🏽
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Video player
    webrtc_streamer(
        key="emotion-ai-stream",
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}
    )
    
    st.markdown('</div>', unsafe_allow_html=True)




# Right Side: Charts
with col2:
    st.markdown("### 📊 Analytics")
    
    with state["lock"]:
        df = pd.DataFrame(list(state["counts"].items()), columns=["Emotion", "Count"])
    
    if df["Count"].sum() > 0:
        fig = px.bar(df, x="Emotion", y="Count", height=420)
        
        fig.update_traces(marker_color='#00C9FF', opacity=0.8)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Start camera to see analytics...")




if auto_update:
    time.sleep(1)
    st.rerun()




# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #888; font-size: 12px;">
    &copy; 2025 Sentiment Neural | v2.1 Stable
</div>
""", unsafe_allow_html=True)


