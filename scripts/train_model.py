from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/processed/train', target_size=(128, 128))
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/data/processed/validation', target_size=(128, 128))

# Train the model
model.fit(train_gen, validation_data=val_gen, epochs=10)

# Save the trained model
model.save('C:/Users/shiva/OneDrive/Desktop/FireDetectionSystem/models/fire_detection_model.keras')
