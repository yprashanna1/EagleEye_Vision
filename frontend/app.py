import streamlit as st
import requests
import os
import cv2
import numpy as np
import uuid

# Page Configuration
st.set_page_config(
    page_title="EagleEye Vision Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Theme
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #F0EAD6;
}
h1 {
    font-size: 3rem;
    color: #FFD700;
    text-shadow: 2px 2px 4px #000;
}
h2 {
    color: #F5DEB3;
}
.stFileUploader > label, .stMarkdown {
    color: #F0EAD6;
}
.stButton>button {
    background: #FFD700 !important;
    color: #0f0c29 !important;
    font-weight: bold;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.5);
}
.frame-container {
    border: 4px solid #FFD700;
    border-radius: 12px;
    padding: 8px;
    background: #1E1E1E;
    max-width: 800px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown("<h1>🚨 EagleEye Vision</h1>", unsafe_allow_html=True)
st.markdown("<h2>Accident Detection – Real‑Time Annotated Playback</h2>", unsafe_allow_html=True)
st.markdown("""
Upload a video feed and watch **EagleEye Vision** detect accidents in real time.
Annotated frames will play below, frame by frame.
""", unsafe_allow_html=True)

# File Uploader
uploaded_file = st.file_uploader(
    "Upload a video file for analysis",
    type=["mp4", "avi", "mov", "mkv"]
)

if uploaded_file:
    temp_video_path = f"temp_{uuid.uuid4().hex}.mp4"
    try:
        # Save upload to unique temp file
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("✅ Video uploaded. Processing...")

        # Setup streaming parameters
        backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000/process_video")
        frame_placeholder = st.empty()
        BOUNDARY = b'--frame\r\n'
        progress_bar = st.progress(0, text="Starting...")

        # Stream request
        with requests.post(backend_url, files={"file": open(temp_video_path, "rb")}, stream=True) as resp:
            if not resp.ok:
                st.error(f"Backend error: {resp.status_code}")
            else:
                buffer = b""
                frame_count = 0

                for chunk in resp.iter_content(chunk_size=1024*512):
                    if chunk:
                        buffer += chunk
                        # Extract complete frames
                        while BOUNDARY in buffer:
                            parts = buffer.split(BOUNDARY)
                            for part in parts[:-1]:
                                header_end = part.find(b'\r\n\r\n')
                                if header_end != -1:
                                    jpg = part[header_end+4:]
                                    if jpg:
                                        arr = np.frombuffer(jpg, np.uint8)
                                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                        if img is not None:
                                            # Convert to RGB for correct colors
                                            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                            frame_placeholder.image(
                                                img_rgb,
                                                channels="RGB",
                                                use_column_width=True
                                            )
                                            frame_count += 1
                                            # Indeterminate progress animation
                                            progress_bar.progress(
                                                min((frame_count % 100) / 100, 1.0),
                                                text=f"Processed {frame_count} frames"
                                            )
                            buffer = parts[-1]

                progress_bar.empty()
                st.success("🎉 Processing complete. Email alert sent if accident detected!")
    except Exception as e:
        st.error(f"🚨 Error: {str(e).split('(')[0]}")
    finally:
        # Always clean up temp file
        if temp_video_path and os.path.exists(temp_video_path):
            os.remove(temp_video_path)
            st.info("🧹 Temporary file cleaned up.")
