import os
import pytest
from fastapi.testclient import TestClient
from backend.app import app

client = TestClient(app)

# Provide local test video path via environment variable or default path
TEST_VIDEO_PATH = os.getenv("TEST_VIDEO_PATH", "./test.mp4")

@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH),
                    reason="test.mp4 file is not available locally. Please provide the video file locally to run this test.")
def test_video_upload():
    with open(TEST_VIDEO_PATH, "rb") as video_file:
        response = client.post("/upload", files={"file": ("test.mp4", video_file)})
    assert response.status_code == 200
    assert response.json()["detail"] == "Upload successful!"

@pytest.mark.skipif(not os.path.exists(TEST_VIDEO_PATH),
                    reason="test.mp4 file is not available locally. Please provide the video file locally to run this test.")
def test_websocket_connection():
    with client.websocket_connect("/ws/annotated?video=test.mp4") as ws:
        data = ws.receive_text()
        assert "status" in data or "error" in data
