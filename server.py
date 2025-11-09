import os, base64, cv2, numpy as np
from flask import Flask, request
from flask_socketio import SocketIO, emit
from ultralytics import YOLO
import face_alignment

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# YOLO Model
yolo = YOLO("models/yolov8n.pt")

# Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Face Landmark model
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cpu')

clients = {}

def decode(img):
    data = img.split(",")[1]
    return cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)

def detect_phone(img):
    res = yolo.predict(img, conf=0.45, verbose=False)[0]
    return any(int(b.cls[0]) == 67 for b in res.boxes)

def head_pose(landmarks):
    # Using nose, eyes, chin → simple pose signal
    nose = landmarks[30]   # center nose
    left_eye = landmarks[36]
    right_eye = landmarks[45]
    # if nose goes left or right compared to eyes midpoint → side looking
    mid_x = (left_eye[0] + right_eye[0]) / 2
    if nose[0] < mid_x - 18:  # threshold
        return "left"
    if nose[0] > mid_x + 18:
        return "right"
    return "center"

@socketio.on("connect")
def connect():
    clients[request.sid] = 0
    emit("message", {"type": "ok", "text": "Connected"})

@socketio.on("frame")
def frame(data):
    sid = request.sid
    warns = clients.get(sid, 0)

    img = decode(data["image"])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        warns += 1
        clients[sid] = warns
        if warns >= 3:
            emit("message", {"type": "kick", "text": "Face Missing"})
            return
        emit("message", {"type": "warn", "text": f"Warning {warns}/3: Face Not Visible"})
        return

    # Landmarks
    preds = fa.get_landmarks(img)
    if preds is None:
        emit("message", {"type": "warn", "text": "Face unclear"})
        return

    lm = preds[0]
    pose = head_pose(lm)

    if pose != "center":
        warns += 1
        clients[sid] = warns
        if warns >= 3:
            emit("message", {"type": "kick", "text": "Looking Away Multiple Times"})
            return
        emit("message", {"type": "warn", "text": f"Warning {warns}/3: Looking {pose}"})
        return

    # Phone detect
    if detect_phone(img):
        warns += 1
        clients[sid] = warns
        if warns >= 3:
            emit("message", {"type":"kick","text":"Phone Detected"})
            return
        emit("message", {"type":"warn","text":f"Warning {warns}/3: Phone Detected"})
        return

    clients[sid] = max(0, warns - 0.1)
    emit("message", {"type":"ok","text":"Monitoring..."}, to=sid)

@app.route("/")
def home():
    return "Backend Running ✅"

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
