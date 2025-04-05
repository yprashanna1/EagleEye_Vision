import streamlit as st
import requests
from io import BytesIO
import cv2
import numpy as np
import os

# Configure page
st.set_page_config(page_title="EagleEye_Vision Dashboard", layout="wide")

# Custom CSS for luxury theme (optional)
st.markdown("""
<style>
body {
    background-color: #1E1E1E;  /* dark background */
    color: #F0EAD6;            /* ivory text for contrast */
}
header, .stButton>button, .stFileUploader {
    background-color: #262730;
    border: none;
}
.stButton>button {
    color: white;
    background-color: #A77D2D !important;  /* gold button */
    border-radius: 5px;
}
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🚨 EagleEye Vision – Accident Detection CCTV Dashboard")
st.markdown(
    "Upload a CCTV video feed and watch **EagleEye_Vision** detect accidents in real time. "
    "Frames with detected accidents will be highlighted and an alert will be sent automatically."
)

# File uploader
uploaded_file = st.file_uploader("Upload a video file for analysis", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    # Write uploaded video to a temporary file
    temp_video_path = "temp_input_video.mp4"
    with open(temp_video_path, "wb") as tempf:
        tempf.write(uploaded_file.getbuffer())
    st.success("Video uploaded. Processing...")

    # Placeholder for video frames
    frame_placeholder = st.empty()
    accident_detected_flag = False

    # Open a connection to the backend streaming endpoint
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000/process_video")    # Stream the video frames from backend
    with requests.post(backend_url, files={"file": open(temp_video_path, "rb")}, stream=True) as response:
        if response.status_code != 200:
            st.error(f"Backend returned an error: {response.status_code}")
        else:
            # The response is a multipart stream of JPEG frames
            bytes_buffer = bytes()
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    bytes_buffer += chunk
                    # Frame boundary for multipart is --frame
                    if b'--frame' in bytes_buffer:
                        # Split at the boundary
                        parts = bytes_buffer.split(b'--frame')
                        # last part is incomplete, keep it for next loop
                        bytes_buffer = parts[-1]
                        # iterate over complete frame parts (skip the last incomplete part)
                        for part in parts[:-1]:
                            if part.startswith(b'\r\nContent-Type: image/jpeg'):
                                # Extract the JPEG image bytes
                                start = part.find(b'\r\n\r\n') + 4
                                jpg_bytes = part[start:]
                                # Convert to numpy array for display
                                frame_array = np.frombuffer(jpg_bytes, np.uint8)
                                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR (cv2) to RGB (streamlit)
                                frame_placeholder.image(frame, channels="RGB")
            # Once stream is done
            st.success("Processing complete.")
    # Optionally, notify if an accident was detected
    # (The backend will handle emailing; here we just inform the user)
    # This flag could be updated via another endpoint or inferred from frames, but for simplicity omitted.
