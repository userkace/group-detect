import cv2
import numpy as np
from tensorflow.keras.models import load_model
import sys
import io

# Redirect stdout to handle Unicode characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
MODEL_PATH = 'face_recognition_model.h5'
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
IMAGE_SIZE = (224, 224)

# Load the trained model
model = load_model(MODEL_PATH)
name_mapping = {
    0: "Dizon",
    1: "Galang",
    2: "Gorospe",
    3: "Padilla"
}

# Initialize face cascade for detection
face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# OpenCV video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(64, 64))

    # if len(faces) == 0:
    #     print("No faces detected.")
    #     continue

    for (x, y, w, h) in faces:
        # Extract the face region and preprocess it
        face_roi = frame[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, IMAGE_SIZE)
        face_roi = np.expand_dims(face_roi, axis=0)
        face_roi = face_roi / 255.0

        # Predict the person
        prediction = model.predict(face_roi)
        predicted_label = np.argmax(prediction[0])
        confidence = prediction[0][predicted_label]

        # Draw rectangle and label on the frame
        color = [(128, 0, 128), (0, 255, 0), (255, 0, 0), (0, 165, 255)][predicted_label]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

        # Add text with confidence
        label_text = f'{name_mapping[predicted_label]} ({confidence*100:.2f}%)'
        cv2.putText(frame, label_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    # Display the frame
    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()