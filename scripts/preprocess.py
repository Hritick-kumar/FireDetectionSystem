import cv2
import os

# Resize and preprocess images
def preprocess_images(input_dir, output_dir, img_size=(128, 128)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            cv2.imwrite(os.path.join(output_dir, img_name), img)
           
# Preprocess Fire and Non-Fire Images
preprocess_images('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/raw/fire', 'C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/processed/fire')
preprocess_images('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/raw/non_fire', 'C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/processed/non_fire')
