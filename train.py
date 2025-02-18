import cv2
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Configuration
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 64
NUM_CLASSES = 4
EPOCHS = 50
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'face_recognition_model.h5'

# Data preprocessing
def load_images_from_directory(directory):
    images = []
    labels = []

    for label, person_dir in enumerate(os.listdir(directory)):
        person_path = os.path.join(directory, person_dir)

        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)

            if image is None:
                print(f"Failed to load {img_path}")
                continue

            image = cv2.resize(image, IMAGE_SIZE)
            images.append(image)
            labels.append(label)

    return np.array(images), np.array(labels)

# Load and preprocess the data
images, labels = load_images_from_directory(DATASET_DIR)
labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)

# Data augmentation
datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

datagen.fit(images)

# Load pre-trained model (MobileNetV2) and add custom layers for classification
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the pre-trained model
for layer in base_model.layers:
    layer.trainable = False

# Compile and train the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(datagen.flow(images, labels, batch_size=BATCH_SIZE), epochs=EPOCHS)

# Save the trained model
model.save(MODEL_SAVE_PATH)