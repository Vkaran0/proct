import os, base64, cv2, numpy as np
from flask import Flask
from flask_socketio import SocketIO, emit
import mediapipe as mp
from ultralytics import YOLO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Load YOLO Model (put yolov8n.pt in models folder or let it auto-download)
MODEL_PATH = "models/yolov8n.pt"
yolo = YOLO(MODEL_PATH)

# MediaPipe FaceMesh
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.5,
                             min_tracking_confidence=0.5)

# Per student warnings
clients = {}   # sid -> warn_count

def decode_frame(b64img):
    data = b64img.split(",")[1]
    img = cv2.imdecode(np.frombuffer(base64.b64decode(data), np.uint8), cv2.IMREAD_COLOR)
    return img

def detect_phone(img):
    res = yolo.predict(img, conf=0.40, verbose=False)[0]
    for box in res.boxes:
        if int(box.cls[0]) == 67:   # class 67 = phone (COCO)
            return True
    return False

@socketio.on("connect")
def on_connect():
    clients[request.sid] = 0
    emit("message", {"type":"ok", "text":"Connected"})

@socketio.on("disconnect")
def on_disconnect():
    clients.pop(request.sid, None)

@socketio.on("frame")
def on_frame(data):
    sid = request.sid
    warn = clients.get(sid, 0)

    img = decode_frame(data["image"])
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)

    # Check face present
    if not res.multi_face_landmarks:
        warn += 1
        clients[sid] = warn
        if warn >= 3:
            emit("message", {"type": "kick", "text": "Face Not Detected 3 Times"}, to=sid)
            return
        emit("message", {"type": "warn", "text": f"Warning {warn}/3: Face Not Visible"}, to=sid)
        return

    # Check phone
    if detect_phone(img):
        warn += 1
        clients[sid] = warn
        if warn >= 3:
            emit("message", {"type": "kick", "text": "Phone Detected (Exam End)"}, to=sid)
            return
        emit("message", {"type": "warn", "text": f"Warning {warn}/3: Phone Detected"}, to=sid)
        return

    # Everything OK
    clients[sid] = max(0, warn - 0.1)   # slow recovery
    emit("message", {"type":"ok","text":"Monitoring..."}, to=sid)

@app.route("/")
def home():
    return "Backend Running ✅"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port)
