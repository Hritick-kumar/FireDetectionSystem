import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/models/fire_detection_model.keras')

# Function to predict fire
def detect_fire(frame, model):
    img = cv2.resize(frame, (128, 128))
    img = np.expand_dims(img, axis=0) / 255.0
    prediction = model.predict(img)
    return 'Fire' if np.argmax(prediction) == 0 else 'No Fire'

# Open camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    label = detect_fire(frame, model)
    color = (0, 0, 255) if label == 'Fire' else (0, 255, 0)
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Fire Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

