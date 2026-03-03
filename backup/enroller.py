import cv2
import torch
from PIL import Image
from torchvision import transforms
import os
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
    print(" error:siamese_model.pth not found, please train the model first")
    exit()

db_file = "face_db.pt"
face_db = {}
if os.path.exists(db_file):
    try:
        face_db = torch.load(db_file)
        print(f"Loaded DB: {list(face_db.keys())}")
    except:
        print("Creating new DB...")

person_name = input("Enter Name: ").strip()
if person_name == "": exit()

if person_name in face_db:
    print(f"Appending new images to {person_name}")


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
emb_list = []

print("Press 'c' to capture, 'q' to finish")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(frame, f"Captured: {len(emb_list)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Enroll - Siamese", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('c'):
        if len(faces) > 0:
            (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
            face_bgr = frame[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(face_rgb)

            try:
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = model.forward_once(img_tensor)
                emb_list.append(emb.cpu())
                print(f"📸 Saved image {len(emb_list)}")
            except Exception as e:
                print(e)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if not emb_list: exit()

if person_name in face_db:
    if isinstance(face_db[person_name], list):
        face_db[person_name].extend(emb_list)
    else:
        face_db[person_name] = [face_db[person_name]] + emb_list
else:
    face_db[person_name] = emb_list

torch.save(face_db, db_file)
print(f" Saved {person_name} to {db_file}")