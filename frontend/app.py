import streamlit as st
import requests
import os
import cv2
import numpy as np

# Configure page
st.set_page_config(page_title="EagleEye Vision Dashboard", layout="wide")
st.title("🚨 EagleEye Vision")
st.markdown(
    "Upload a video feed and watch **EagleEye Vision** detect accidents in real time. "
    "Annotated frames will be displayed below."
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
        # Send the video file to the backend with streaming enabled
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
                            # Process all complete parts (all parts except the last, which may be incomplete)
                            for part in parts[:-1]:
                                part = part.strip()
                                if not part:
                                    continue
                                # Find the header terminator (the blank line after headers)
                                header_end = part.find(b'\r\n\r\n')
                                if header_end == -1:
                                    continue
                                jpg_bytes = part[header_end+4:]
                                if not jpg_bytes:
                                    continue
                                # Decode the JPEG image
                                frame_array = np.frombuffer(jpg_bytes, np.uint8)
                                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
                                if frame is not None:
                                    # Convert from BGR to RGB for display in Streamlit
                                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    frame_placeholder.image(frame, channels="RGB")
                            # Retain the last incomplete part for the next iteration
                            bytes_buffer = parts[-1]
        st.success("Processing complete.")
    except Exception as e:
        st.error(f"Error during streaming: {e}")
