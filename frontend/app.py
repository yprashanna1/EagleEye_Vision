import streamlit as st
import requests
import os
import cv2
import numpy as np

# Configure page
st.set_page_config(page_title="EagleEye Vision Dashboard", layout="wide")

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
st.title("🚨 EagleEye Vision")
st.markdown(
    "Upload a video feed and watch **EagleEye Vision** detect accidents in real time. "
    "Frames with detected accidents will be highlighted and an alert will be sent automatically."
)

# File uploader widget
uploaded_file = st.file_uploader("Upload a video file for analysis", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    # Write uploaded video to a temporary file
    temp_video_path = "temp_input_video.mp4"
    with open(temp_video_path, "wb") as tempf:
        tempf.write(uploaded_file.getbuffer())
    st.success("Video uploaded. Processing...")

    # Get backend URL from environment variable; default to localhost for testing
    backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000/process_video")
    
    # Create a placeholder for video frames
    frame_placeholder = st.empty()
    
    try:
        with requests.post(backend_url, files={"file": open(temp_video_path, "rb")}, stream=True) as response:
            if response.status_code != 200:
                st.error(f"Backend returned an error: {response.status_code}")
            else:
                bytes_buffer = b""
                # Continuously read the stream in chunks
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        bytes_buffer += chunk
                        # Process as long as we have a complete frame
                        while b'--frame' in bytes_buffer:
                            parts = bytes_buffer.split(b'--frame')
                            # The last part may be incomplete; keep it for next iteration
                            if len(parts) > 1:
                                frame_part = parts[1]
                                bytes_buffer = b'--frame'.join(parts[2:])
                                # Check if we have the complete header and JPEG data
                                header_end = frame_part.find(b'\r\n\r\n')
                                if header_end != -1:
                                    jpg_bytes = frame_part[header_end+4:]
                                    # Decode the JPEG image
                                    frame_array = np.frombuffer(jpg_bytes, np.uint8)
                                    frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                    if frame is not None:
                                        # Convert BGR to RGB for proper color display
                                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(frame, channels="RGB")
        st.success("Processing complete.")
    except Exception as e:
        st.error(f"Error during streaming: {e}")
