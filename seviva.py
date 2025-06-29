import os
import io
import cv2
import numpy as np
import warnings
import calendar
from datetime import datetime
from flask import Flask, request, send_file, Response
import torch
import torch.nn as nn
from torchvision import transforms
from flask_cors import CORS
import threading
import time
import firebase_admin
from firebase_admin import credentials, db, storage

app = Flask(__name__)
CORS(app)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Inicializar Firebase ===
if not firebase_admin._apps:
    cred = credentials.Certificate("serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {
        "databaseURL": "https://system-detect-default-rtdb.firebaseio.com/",
        "storageBucket": "system-detect.firebasestorage.app"
    })

bucket = storage.bucket()

# === Modelo CNN ===
class DetectorCNN(nn.Module):
    def __init__(self):
        super(DetectorCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 12 * 12, 128), nn.ReLU(),
            nn.Linear(128, 1), nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# === Configuración del modelo ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DetectorCNN().to(device)
model_path = os.path.join(os.path.dirname(__file__), "human_cnn_final.pth")
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Transformaciones de imagen ===
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((96, 96)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# === Control de background y sincronización ===
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
ultimo_reinicio = time.time()
latest_frame = None
frame_lock = threading.Lock()

def get_15s_block(now: datetime) -> str:
    second = (now.second // 15) * 15
    return f"{now.hour:02d}-{now.minute:02d}-{second:02d}"

@app.route("/")
def index():
    return "Servidor activo."

@app.route("/process_frame", methods=["POST"])
def process_frame():
    global latest_frame, bg_subtractor, ultimo_reinicio

    file = request.files.get("frame")
    if not file:
        return "No frame received", 400

    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    if frame is None:
        return "Invalid image", 400

    ahora = datetime.now()
    timestamp = ahora.strftime("%Y%m%d_%H%M%S_%f")
    fecha = ahora.strftime("%Y-%m-%d")
    dia = calendar.day_name[ahora.weekday()]
    bloque = get_15s_block(ahora)

    fg_mask = bg_subtractor.apply(frame)
    fg_mask = cv2.medianBlur(fg_mask, 5)
    fg_mask = cv2.dilate(fg_mask, None, iterations=2)

    if np.sum(fg_mask == 255) > frame.shape[0] * frame.shape[1] * 0.6 and (time.time() - ultimo_reinicio > 2):
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        ultimo_reinicio = time.time()
        return "", 204

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = frame.copy()
    humanos_detectados = 0
    confianza_maxima = 0

    for cnt in contours:
        if cv2.contourArea(cnt) < 10000:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        recorte = frame[y:y+h, x:x+w]
        if recorte.size == 0:
            continue

        tensor_img = transform(recorte).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(tensor_img).item()
            print(f"[DEBUG] pred: {pred:.4f}")

        confianza_maxima = max(confianza_maxima, pred)

        if pred >= 0.98:
            humanos_detectados += 1
            color = (0, 255, 0)
            label = f"Humano ({pred:.2f})"
            cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)
            cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(result, ahora.strftime("%Y-%m-%d %H:%M:%S"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(result, f"Personas: {humanos_detectados}", (result.shape[1] - 160, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    cv2.putText(result, "verde >= 0.98", (10, result.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    _, img_encoded = cv2.imencode('.jpg', result)
    with frame_lock:
        latest_frame = img_encoded.tobytes()

    if humanos_detectados > 0:
        deteccion_dir = os.path.join("frames_guardados", "detecciones", f"{dia}_{fecha}", bloque)
        os.makedirs(deteccion_dir, exist_ok=True)

        filename = f"deteccion_{timestamp}.jpg"
        local_path = os.path.join(deteccion_dir, filename)
        cv2.imwrite(local_path, result)

        blob = bucket.blob(f"images/detecciones/{dia}_{fecha}/{bloque}/{filename}")
        blob.upload_from_filename(local_path)
        blob.make_public()
        image_url = blob.public_url

        try:
            db.reference("/detecciones").push({
                "timestamp": ahora.isoformat(),
                "dia": dia,
                "fecha": fecha,
                "bloque": bloque,
                "url_imagen": image_url,
                "personas_detectadas": humanos_detectados,
                "confianza_maxima": round(confianza_maxima, 4)
            })
            print("[Firebase] Imagen subida y registrada.")
        except Exception as e:
            print(f"[Firebase ERROR] {e}")

        return send_file(io.BytesIO(img_encoded.tobytes()), mimetype='image/jpeg')

    return "", 204

@app.route("/stream")
def stream():
    def generate():
        while True:
            with frame_lock:
                if latest_frame:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + latest_frame + b"\r\n")
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)