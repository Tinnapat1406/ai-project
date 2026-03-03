import cv2

cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if cascade.empty():
    print("Error loading Haar Cascade XML file")
else:
    print("Loaded Haar Cascade XML file successfully")
