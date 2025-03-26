# backend/tests/test_video.py
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

def test_video_upload():
    response = client.post("/upload", files={"file": ("test.mp4", open("test.mp4", "rb"))})
    assert response.status_code == 200
    assert response.json()["detail"] == "Upload successful!"

def test_websocket_connection():
    ws = client.websocket_connect("/ws/annotated?video=test.mp4")
    data = ws.receive_text()
    assert "status" in data or "error" in data
