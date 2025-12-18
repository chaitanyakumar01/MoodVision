import cv2
from deepface import DeepFace

# Project: Real-time Emotion Detector
# Description: This script uses the webcam to detect faces and analyze emotions.
# Note: DeepFace might be a bit slow on CPU, but it is very accurate.

# -----------------------------------------------------------
# 1. CONFIGURATION: Colors and Emojis
# OpenAI uses BGR format (Blue, Green, Red) instead of RGB.
# So, Red is (0, 0, 255) and Blue is (255, 0, 0).
# -----------------------------------------------------------
emotion_style = {
    'happy':    {'color': (0, 255, 255), 'emoji': 'Happy :)'},      # Yellow (Blue+Green)
    'sad':      {'color': (255, 0, 0),   'emoji': 'Sad :('},        # Blue
    'angry':    {'color': (0, 0, 255),   'emoji': 'Angry >:@'},     # Red
    'surprise': {'color': (0, 255, 0),   'emoji': 'Wow :O'},        # Green
    'neutral':  {'color': (200, 200, 200), 'emoji': 'Normal :|'},   # Light Grey
    'fear':     {'color': (128, 0, 128), 'emoji': 'Scared o_O'},    # Purple
    'disgust':  {'color': (0, 128, 255), 'emoji': 'Yuck XP'}        # Orange
}

# -----------------------------------------------------------
# 2. INITIALIZE CAMERA
# '0' usually refers to the default webcam built into the laptop.
# -----------------------------------------------------------
print("System: Initializing Camera... Please wait...")
cap = cv2.VideoCapture(0)

# Safety Check: Sometimes another app might be using the camera
if not cap.isOpened():
    print("❌ Error: Could not open the webcam. Please check your settings.")
    exit()

print("✅ System Ready! Camera is live. Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    # 'ret' is a boolean (True/False) indicating if the frame was read correctly
    # 'frame' is the actual image captured
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Selfie Mode: 
    # By default, webcams are not mirrored. This flips the image horizontally (1)
    # so it feels like looking into a mirror.
    frame = cv2.flip(frame, 1)

    try:
        # -------------------------------------------------------
        # CORE LOGIC: DeepFace Analysis
        # -------------------------------------------------------
        # actions=['emotion'] -> We only want emotion, not age or gender.
        # enforce_detection=False -> VERY IMPORTANT! 
        # If this is True (default), the code will CRASH if no face is found.
        # We set it to False so it just returns nothing instead of crashing.
        analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        # Iterate through all detected faces (in case there are multiple people)
        for face in analysis:
            # Extracting the coordinates of the face box (Region of Interest)
            x = face['region']['x']
            y = face['region']['y']
            w = face['region']['w'] # Width
            h = face['region']['h'] # Height
            
            # Get the emotion with the highest score
            current_mood = face['dominant_emotion']
            
            # Fetch color and emoji from our settings dictionary
            if current_mood in emotion_style:
                style = emotion_style[current_mood]
                box_color = style['color']
                display_text = style['emoji']
            else:
                # Fallback in case the model predicts something unexpected
                box_color = (255, 255, 255)
                display_text = current_mood

            # ---------------------------------------------------
            # DRAWING ON THE SCREEN
            # ---------------------------------------------------
            
            # 1. Draw the bounding box around the face
            # Arguments: Image, Start Point (x,y), End Point (x+w, y+h), Color, Thickness
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            
            # 2. Draw the Text (Mood + Emoji)
            # To make the text readable on any background, we draw it twice:
            # First layer: A thick BLACK line (acts as a border/shadow)
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            # Second layer: The colored text on top
            cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)

    except Exception as e:
        # If DeepFace fails or something glitchy happens, print error but DON'T stop the video
        # print(f"Warning: {e}") 
        pass

    # -----------------------------------------------------------
    # DISPLAY OUTPUT
    # -----------------------------------------------------------
    # Add an instruction text at the top left corner
    cv2.putText(frame, "Press 'Q' to Exit", (10, 30), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    # Show the final frame in a window titled "My Emotion Project"
    cv2.imshow('My Emotion Project', frame)

    # Quit Logic:
    # cv2.waitKey(1) waits 1ms for a key press.
    # & 0xFF ensures it works correctly on 64-bit machines.
    # ord('q') returns the ASCII value of 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("System: Closing Application... Goodbye!")
        break

# -----------------------------------------------------------
# CLEANUP
# Release the camera resource and close all open windows
# -----------------------------------------------------------
cap.release()
cv2.destroyAllWindows()