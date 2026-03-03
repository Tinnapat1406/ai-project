import cv2
import csv
import os
import datetime
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from model import SiameseEfficientNet


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print(f"Loading Siamese Model on {device}...")
model = SiameseEfficientNet().to(device)

if os.path.exists("siamese_model.pth"):
    model.load_state_dict(torch.load("siamese_model.pth", map_location=device))
    model.eval()
else:
    print("error:siamese_model.pth not found, please train the model first")
    exit()


if os.path.exists("face_db.pt"):
    face_db = torch.load("face_db.pt", map_location=device)
    print("Loaded face_db:", list(face_db.keys()))

    for name in face_db:
        face_db[name] = face_db[name].to(device)
else:
    print("Error: no face_db.pt found,starting with empty database")
    exit()

csv_file = "attendance.csv"
file_exists = os.path.isfile(csv_file)

f = open(csv_file, "a", newline="", encoding="utf-8-sig")
writer = csv.writer(f)

if not file_exists:
    writer.writerow(["name", "datetime", "status"])

marked_names = set()


def identify_face(pil_img, threshold=0.8): 
    img_tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        q_emb = model.forward_once(img_tensor)

    best_name = "unknown"
    best_dist = 999.0

    for name, db_emb in face_db.items():
        dist = F.pairwise_distance(q_emb, db_emb.unsqueeze(0)).item()
        
        if dist < best_dist:
            best_dist = dist
            best_name = name
    if best_dist > threshold:
        return "unknown", best_dist
    
    return best_name, best_dist


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    f.close()
    exit()

print("starting video stream... press 'q' to quit")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_bgr = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)

        name, dist = identify_face(pil_img, threshold=0.8)
        color = (0, 255, 0) if name != "unknown" else (0, 0, 255)


        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{name} ({dist:.2f})"
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if name != "unknown" and name not in marked_names:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([name, now, "check-in"])
            f.flush() 
            marked_names.add(name)
            print(f"[CHECK-IN] {name} datetime {now} (Dist: {dist:.4f})")

    cv2.imshow("Face Attendance Siamese", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
f.close()
cv2.destroyAllWindows()