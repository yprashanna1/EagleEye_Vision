from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
from ultralytics import YOLO
import threading, os, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
from collections import deque

app = FastAPI()

# Add CORS middleware (for production, replace "*" with your actual frontend URL)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv8 model at startup (assumes best.pt is in the ./model directory)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "best.pt")
model = YOLO(MODEL_PATH)

@app.post("/process_video")
def process_video(file: UploadFile = File(...)):
    """
    Accepts a video file, streams back annotated frames, and if an accident is detected,
    saves a short clip and automatically sends an email with the clip attached.
    """
    # Save uploaded video to a temporary file
    temp_filename = "temp_upload_video.mp4"
    with open(temp_filename, "wb") as f:
        for chunk in file.file:
            f.write(chunk)
    file.file.close()

    # Open the video file with OpenCV
    cap = cv2.VideoCapture(temp_filename)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0  # fallback FPS is 24 if not available
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    # Prepare for accident clip capturing
    pre_event_buffer = deque(maxlen=int(fps * 5))  # buffer last 5 seconds of frames
    clip_frames = []  # to store frames for the accident clip
    accident_detected = False
    post_event_frames_to_capture = 0

    def frame_generator():
        nonlocal accident_detected, post_event_frames_to_capture
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                # Run YOLOv8 inference on the frame (on CPU)
                results = model.predict(frame, conf=0.4, device="cpu", verbose=False)
                result = results[0]
                detections = result.boxes  # Get detection boxes
                if detections and len(detections) > 0:
                    # For each detected object, assume class 0 = "Accident"
                    for box in detections:
                        cls_id = int(box.cls[0]) if hasattr(box, 'cls') else 0
                        conf = float(box.conf[0]) if hasattr(box, 'conf') else 0.0
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
                        label = f"Accident {conf*100:.1f}%"
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    if not accident_detected:
                        accident_detected = True
                        # Save the previous 5 seconds of frames (pre-event)
                        clip_frames.extend(list(pre_event_buffer))
                        # Save the current frame as the start of the clip
                        clip_frames.append(frame.copy())
                        # Set to capture next 5 seconds of frames after detection
                        post_event_frames_to_capture = int(fps * 5)
                    else:
                        if post_event_frames_to_capture > 0:
                            clip_frames.append(frame.copy())
                            post_event_frames_to_capture -= 1
                else:
                    # No detection: annotate as Non-Accident
                    cv2.putText(frame, "Non-Accident", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    if accident_detected and post_event_frames_to_capture > 0:
                        clip_frames.append(frame.copy())
                        post_event_frames_to_capture -= 1

                # Add current frame to pre-event buffer (maintaining last 5 seconds)
                pre_event_buffer.append(frame.copy())

                # Encode frame as JPEG
                success, encoded_image = cv2.imencode(".jpg", frame)
                if not success:
                    continue
                frame_bytes = encoded_image.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        finally:
            cap.release()
            # After processing the video, if an accident was detected, save the clip and send email
            if accident_detected and clip_frames:
                clip_path = "accident_clip.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))
                for f in clip_frames:
                    out_writer.write(f)
                out_writer.release()
                # Send email in a background thread so as not to delay the response
                threading.Thread(target=send_email_with_clip, args=(clip_path,)).start()
            # Remove temporary upload file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)

    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

def send_email_with_clip(clip_path):
    gmail_user = os.environ.get("GMAIL_USER")
    gmail_pass = os.environ.get("GMAIL_PASS")
    recipient = os.environ.get("RECIPIENT_EMAIL")
    
    if not gmail_user or not gmail_pass or not recipient:
        print("Email credentials or recipient not set in environment.")
        return
    
    subject = "Accident Detected by EagleEye_Vision"
    body = (
        "Dear User,\n\nAn accident was detected by EagleEye_Vision. "
        "See the attached video clip for details.\n\nRegards,\nEagleEye_Vision Alert"
    )

    msg = MIMEMultipart()
    msg['From'] = gmail_user
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(clip_path, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', f'attachment; filename="accident_clip.mp4"')
    msg.attach(part)

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(gmail_user, gmail_pass)
        server.sendmail(gmail_user, recipient, msg.as_string())
        server.quit()
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
