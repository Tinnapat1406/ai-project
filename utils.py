import os
import torch
from torchvision import transforms
from PIL import Image
from model import SiameseEfficientNet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


inference_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path="siamese_model.pth"):
    print(f"Loading model from {model_path} on {device}...")
    model = SiameseEfficientNet().to(device)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        

        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            
            print("Warning: Key mismatch, loading with strict=False")
            model.load_state_dict(state_dict, strict=False)
            
        model.eval() 
        return model
    else:
        print(f" Error: Model file not found at {model_path}")
        return None

def load_face_db(db_path="face_db.pt"):       
    if os.path.exists(db_path):
        print(f"Loading face_db from {db_path}")
        db = torch.load(db_path, map_location=device)
        return db
    else:
        return {}

def save_face_db(face_db, db_path="face_db.pt"):
    cpu_db = {name: emb.cpu() for name, emb in face_db.items()}
    torch.save(cpu_db, db_path)
    print(f"Saved face_db to {db_path}")

def get_embedding_from_pil(img_pil, model):

    img_tensor = inference_transform(img_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.forward_once(img_tensor)

    return embedding.squeeze(0)