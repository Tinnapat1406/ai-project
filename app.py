
from flask import Flask, render_template, Response, request, jsonify
import cv2
import datetime
import csv
import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import base64
import io
from collections import Counter

from model import SiameseEfficientNet  

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on Device: {device}")

transform = transforms.Compose(
    [
        transforms.Resize((160, 160)),
        transforms.ToTensor(),  
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  
    ]
)

model = SiameseEfficientNet().to(device)
model.eval()
print(" Loaded FaceNet backbone (InceptionResnetV1, pretrained)")

face_db = {}
if os.path.exists("face_db.pt"):
    try:
        loaded_db = torch.load("face_db.pt", map_location=device)
        for name, data in loaded_db.items():
            if isinstance(data, list):
                face_db[name] = [emb.to(device) for emb in data]
            else:
                face_db[name] = [data.to(device)]
        print(f"Loaded face_db: {len(face_db)} users")
    except Exception as e:
        print(f"Error loading face_db: {e}")

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#Check-in cooldown tracker Logic
last_checkin = {}  

def can_checkin(name, cooldown_minutes=10):

    now = datetime.datetime.now()
    last = last_checkin.get(name)

    if last is None:
        last_checkin[name] = now
        return True, now

    diff_sec = (now - last).total_seconds()
    if diff_sec >= cooldown_minutes * 60:
        last_checkin[name] = now
        return True, now
    return False, last


#Identify 
def identify_face(pil_img, threshold=0.8):

    if model is None:
        return "System Error", 999.0

    img_tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        q_emb = model.forward_once(img_tensor)

    best_name = "unknown"
    best_dist = 999.0

    for name, db_embs in face_db.items():
        for db_emb in db_embs:
            dist = F.pairwise_distance(q_emb, db_emb.unsqueeze(0)).item()
            if dist < best_dist:
                best_dist = dist
                best_name = name

    if best_dist > threshold:
        return "unknown", best_dist
    return best_name, best_dist


#Live mode
def gen_frames():
    cap = cv2.VideoCapture(0)

    csv_file = "attendance.csv"
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["name", "datetime", "status"])

    history = []
    max_history = 7 

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            if len(history) > 0:
                history.pop(0)

        for x, y, w, h in faces:
            face_bgr = frame[y : y + h, x : x + w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            raw_name, dist = identify_face(pil_img, threshold=0.8)

            history.append(raw_name)
            if len(history) > max_history:
                history.pop(0)

            final_name = raw_name
            if history:
                count = Counter(history)
                best, vote = count.most_common(1)[0]
                if vote >= (len(history) // 2) + 1:
                    final_name = best
                else:
                    final_name = "unknown"

            color = (0, 255, 0) if final_name != "unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(
                frame,
                f"{final_name} ({dist:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if final_name != "unknown":
                ok, check_time = can_checkin(final_name, cooldown_minutes=10)
                if ok:
                    now_str = check_time.strftime("%Y-%m-%d %H:%M:%S")
                    with open(
                        "attendance.csv", "a", newline="", encoding="utf-8-sig"
                    ) as f:
                        csv.writer(f).writerow([final_name, now_str, "check-in"])
                    print(f"[Check-in] {final_name} datetime {now_str}")
                else:
                    print(f"[Cooldown]  {final_name} cannot check-in yet (last at {check_time.strftime('%H:%M:%S')})")

        ret, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )

    cap.release()


def scan_once(num_frames=15, threshold=0.8):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam in scan_once")
        return None, None

    history = []
    dist_history = []

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            continue

        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])

        face_bgr = frame[y : y + h, x : x + w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        raw_name, dist = identify_face(pil_img, threshold=threshold)
        history.append(raw_name)
        dist_history.append(dist)

    cap.release()

    if not history:
        print("No faces detected during scan")
        return "unknown", None

  
    count = Counter(history)
    best_name, votes = count.most_common(1)[0]

    if best_name == "unknown":
        return "unknown", min(d for d in dist_history if d is not None)

    req_ratio = 0.7
    if votes < int(len(history) * req_ratio):
        print(f"Not confident: {best_name} received {votes}/{len(history)} votes")
        return "unknown", min(
            d for d, n in zip(dist_history, history) if n == best_name
        )

    best_dist = min(d for d, n in zip(dist_history, history) if n == best_name)

    print(
        f"[SCAN ONCE RESULT] name={best_name}, dist={best_dist:.4f}, votes={votes}/{len(history)}"
    )
    return best_name, best_dist


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/logs")
def logs():
    rows = []
    if os.path.exists("attendance.csv"):
        with open("attendance.csv", encoding="utf-8-sig") as f:
            rows = list(csv.reader(f))
    return render_template("logs.html", rows=rows)


@app.route("/enroll")
def enroll():
    return render_template("enroll.html")


@app.route("/scan")
def scan():
    """Render a simple Scan page (placeholder)
    """
    return render_template("scan.html")


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    global face_db
    data = request.get_json()
    name = data.get("name")
    img_data = data.get("image")

    if not name or not img_data:
        return jsonify({"ok": False}), 400

    try:
        header, encoded = img_data.split(",", 1)
        pil_img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = model.forward_once(img_tensor)

        if name in face_db:
            face_db[name].append(emb.to(device))
        else:
            face_db[name] = [emb.to(device)]

        cpu_db = {}
        for k, v_list in face_db.items():
            cpu_db[k] = [e.cpu() for e in v_list]
        torch.save(cpu_db, "face_db.pt")

        return jsonify({"ok": True, "message": f"Saved {name}"})
    except Exception as e:
        print(e)
        return jsonify({"ok": False}), 500

@app.route("/api/checkin_once", methods=["POST"])
def api_checkin_once():
    name, dist = scan_once(num_frames=15, threshold=0.8)

    if name is None:
        return jsonify({"ok": False, "error": "camera_error"}), 500

    if name == "unknown" or dist is None:
        return jsonify({"ok": False, "name": "unknown"}), 200

    csv_file = "attendance.csv"
    if not os.path.isfile(csv_file):
        with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
            csv.writer(f).writerow(["name", "datetime", "status"])

    ok, check_time = can_checkin(name, cooldown_minutes=10)

    if not ok:
        return jsonify(
            {"ok": True, "name": name, "dist": round(dist, 3), "message": "cooldown"}
        )

    now_str = check_time.strftime("%Y-%m-%d %H:%M:%S")
    with open(csv_file, "a", newline="", encoding="utf-8-sig") as f:
        csv.writer(f).writerow([name, now_str, "check-in"])
    print(f"[Scan Once Check-in] {name} datetime {now_str}")

    return jsonify(
        {"ok": True, "name": name, "dist": round(dist, 3), "message": "new_checkin"}
    )


@app.route("/api/list_faces", methods=["GET"])
def api_list_faces():
    return jsonify({"ok": True, "names": sorted(face_db.keys())})


@app.route("/api/delete_face", methods=["POST"])
def api_delete_face():
    global face_db
    data = request.get_json()
    name = (data.get("name") or "").strip()
    if name in face_db:
        del face_db[name]
        cpu_db = {}
        for k, v_list in face_db.items():
            cpu_db[k] = [e.cpu() for e in v_list]
        torch.save(cpu_db, "face_db.pt")
        return jsonify({"ok": True})
    return jsonify({"ok": False}), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
