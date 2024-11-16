import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Function to test a single image
def test_image(img_path, model):
    # Read the image
    img = cv2.imread(img_path)
    
    # Check if the image was loaded successfully
    if img is None:
        print(f"Error: Unable to load image at {img_path}")
        return
    
    # Resize the image to match the model input
    img_resized = cv2.resize(img, (128, 128))
    
    # Normalize the image (if needed, based on your model)
    img_resized = img_resized / 255.0
    
    # Reshape the image to match the model input shape
    img_reshaped = np.expand_dims(img_resized, axis=0)

    # Make a prediction
    prediction = model.predict(img_reshaped)

    # Print the result
    label = 'Fire' if np.argmax(prediction) == 0 else 'No Fire'
    print(f"Prediction for {img_path}: {label}: {prediction}")

# Load the trained model
model = load_model('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/models/fire_detection_model.keras')

# Test the image (use absolute path for the image)
test_image('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/test/fire_sample.jpg', model)
