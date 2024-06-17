import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
df_train = pd.read_csv('train_final.csv')
labels = df_train.pop('784').values  # Extract labels and remove the label column
images = df_train.values.reshape(-1, 28, 28, 1)  # Reshape data for "channels_last"
labels = to_categorical(labels, num_classes=13)  # Convert labels to one-hot encoding

# Set the random seed for reproducibility
np.random.seed(1212)
tf.random.set_seed(1212)

# Split data into training and validation sets
images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.2, random_state=1212)

# Setting up data augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator()  # No augmentation for validation data

# Create data generators
train_generator = train_datagen.flow(images_train, labels_train, batch_size=200)
val_generator = val_datagen.flow(images_val, labels_val, batch_size=200)

# Define the model architecture
model = Sequential([
    Input(shape=(28, 28, 1)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(13, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model
model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[early_stopping],
    verbose=1
)

# Save the model architecture and weights
model_json = model.to_json()
with open("model_final.json", "w") as json_file: # Will be used for CNN testing
    json_file.write(model_json)
model.save_weights("model_final.weights.h5") # also used for testing

print("Model trained and saved successfully.") 
