import base64, cv2, numpy as np, time
from flask import Flask
from flask_cors import CORS
from flask_socketio import SocketIO, Namespace, emit
import mediapipe as mp

# Flask + Socket
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# Face mesh (for head pose + eyes)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=2, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# YOLO (phone detection)
from ultralytics import YOLO
yolo = YOLO("yolov8n.pt")
PHONE_CLASS_ID = 67

# Face landmark indexes
IDX_NOSE = 1
IDX_CHIN = 152
IDX_EYE_LEFT_OUT = 33
IDX_EYE_RIGHT_OUT = 263
IDX_MOUTH_LEFT = 61
IDX_MOUTH_RIGHT = 291

# Eye gaze indexes
IDX_LEFT_IRIS = 468
IDX_RIGHT_IRIS = 473
IDX_LEFT_IN = 133
IDX_RIGHT_IN = 362

MODEL_POINTS = np.array([
    (0, 0, 0),
    (0, -63, -12),
    (-43, 32, -26),
    (43, 32, -26),
    (-28, -28, -24),
    (28, -28, -24)
], dtype=np.float64)

def head_pose(lm, w, h):
    pts = np.array([
        (lm[IDX_NOSE].x*w, lm[IDX_NOSE].y*h),
        (lm[IDX_CHIN].x*w, lm[IDX_CHIN].y*h),
        (lm[IDX_EYE_LEFT_OUT].x*w, lm[IDX_EYE_LEFT_OUT].y*h),
        (lm[IDX_EYE_RIGHT_OUT].x*w, lm[IDX_EYE_RIGHT_OUT].y*h),
        (lm[IDX_MOUTH_LEFT].x*w, lm[IDX_MOUTH_LEFT].y*h),
        (lm[IDX_MOUTH_RIGHT].x*w, lm[IDX_MOUTH_RIGHT].y*h)
    ], dtype=np.float64)
    cam = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]],dtype=np.float64)
    success, rvec, tvec = cv2.solvePnP(MODEL_POINTS, pts, cam, np.zeros((4,1)))
    R,_ = cv2.Rodrigues(rvec)
    yaw = np.degrees(np.arctan2(-R[2][0], np.sqrt(R[0][0]**2+R[1][0]**2)))
    return yaw

def gaze(lm,w,h):
    def get(center, inner, outer):
        c, i, o = lm[center], lm[inner], lm[outer]
        return ((c.x*w)-(o.x*w)) / ((i.x*w)-(o.x*w)+1e-6)
    g = (get(IDX_LEFT_IRIS,IDX_LEFT_IN,IDX_EYE_LEFT_OUT)+ (1-get(IDX_RIGHT_IRIS,IDX_RIGHT_IN,IDX_EYE_RIGHT_OUT)))/2
    return g

class WarningState:
    def __init__(self):
        self.warn = 0

    def add(self, reason):
        self.warn += 1
        if self.warn >= 4:
            return ("KICK", reason)
        else:
            return ("WARN", f"⚠ Warning {self.warn}: {reason}")

class Stream(Namespace):
    def on_connect(self):
        self.state = WarningState()
        emit("msg", {"type":"ok","text":"Connected"})

    def on_frame(self,data):
        img_data = data["img"].split(",")[1]
        frame = cv2.imdecode(np.frombuffer(base64.b64decode(img_data), np.uint8), cv2.IMREAD_COLOR)
        h,w=frame.shape[:2]
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        res=face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            typ,msg=self.state.add("Face missing")
        else:
            lm=res.multi_face_landmarks[0].landmark

            # Head turn check
            yaw=head_pose(lm,w,h)
            if abs(yaw)>25:
                typ,msg=self.state.add("Looking away from screen")
                if typ=="KICK": emit("msg",{"type":"kick","text":"Exam Terminated"}); return
                emit("msg",{"type":"warn","text":msg}); return

            # Eye gaze check
            g=gaze(lm,w,h)
            if g<0.35 or g>0.65:
                typ,msg=self.state.add("Suspicious eye movement")
                if typ=="KICK": emit("msg",{"type":"kick","text":"Exam Terminated"}); return
                emit("msg",{"type":"warn","text":msg}); return

        # Phone detect
        small=cv2.resize(frame,(480,int(h*(480/w))))
        det=yolo.predict(small,verbose=False)
        for r in det:
            for b in r.boxes:
                if int(b.cls)==PHONE_CLASS_ID:
                    typ,msg=self.state.add("Phone detected")
                    if typ=="KICK": emit("msg",{"type":"kick","text":"Exam Terminated"}); return
                    emit("msg",{"type":"warn","text":msg}); return

        emit("msg",{"type":"ok","text":"Monitoring..."})

socketio.on_namespace(Stream('/stream'))

if __name__=="__main__":
    socketio.run(app,host="0.0.0.0",port=5000)
