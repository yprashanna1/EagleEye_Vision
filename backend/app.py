import os
import json
import asyncio
import cv2
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from backend.detection_model import process_frame, classify_detections
from starlette.websockets import WebSocketState


# Get environment variables
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")  # Get upload path from environment variables
PORT = int(os.getenv("PORT", 8000))  # Get the port from environment variables (default 8000)

app = FastAPI(title="Scalable CCTV Video Detection", version="1.0.0")

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allows all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# create upload directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# route for uploading video files
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    # check extension of uploaded file
    if not filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are allowed.")
    
    # define path where video will be saved
    save_path = os.path.join(UPLOAD_DIR, filename)
    
    # save video to disk
    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)
    
    return {"filename": filename, "detail": "Upload successful!"}

# WebSocket endpoint for annotated frames
@app.websocket("/ws/annotated")
async def websocket_annotated(websocket: WebSocket, video: str = Query(...)):
    await websocket.accept()
    
    # construct path to uploaded video
    video_path = os.path.join(UPLOAD_DIR, video)
    
    # check if video exists or not
    if not os.path.exists(video_path):
        await websocket.send_text(json.dumps({"error": f"Video not found: {video}"}))
        await websocket.close()
        return

    # Open video file using OpenCV
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await websocket.send_text(json.dumps({"error": f"Cannot open video file: {video_path}"}))
        await websocket.close()
        return

    try:
        while True:
            # read video frames asynchronously
            ret, frame = await asyncio.to_thread(cap.read)
            
            # if the frame is not valid, it means the video has ended
            if not ret:
                await websocket.send_text(json.dumps({"status": "Video ended"}))
                break
            
            # process frame to get detections
            detections = process_frame(frame)
            
            # classify detections (includes suspicious and accident detections)
            labeled_detections = classify_detections(detections)
            
            # annotate frame with bounding boxes and labels
            annotated_frame = frame.copy()
            for det, color, label_text in labeled_detections:
                bbox = det["bounding_box"]
                x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
                
                # draw bounding box and label on frame
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, label_text, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # encode annotated frame as JPEG
            success, encoded_image = cv2.imencode(".jpg", annotated_frame)
            if not success:
                await websocket.send_text(json.dumps({"error": "Failed to encode frame"}))
                continue

            # Check if WebSocket is still open before sending data
            if websocket.client_state == WebSocketState.CONNECTED:
                # send frame over WebSocket connection
                await websocket.send_bytes(encoded_image.tobytes())
            else:
                # If WebSocket is closed, stop sending data
                break
            
            # sleep to achieve ~30 FPS
            await asyncio.sleep(0.033)
    except WebSocketDisconnect:
        print("[INFO] Client disconnected.")
    finally:
        # Release video capture when done
        cap.release()
        # Only close the WebSocket if it's still open
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close()


if __name__ == "__main__":
    import uvicorn
    # Run app with Uvicorn on host 0.0.0.0, port 8000, and with auto-reload
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)  # No reload for production
