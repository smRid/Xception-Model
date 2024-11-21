import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the CSV file
csv_path = 'path/to/your/labels.csv'  # Update this path
data_df = pd.read_csv(csv_path)

# Parameters
image_size = (299, 299)
batch_size = 32
num_classes = len(data_df['label'].unique())

# Load images and labels
images = []
labels = []

for _, row in data_df.iterrows():
    img_path = os.path.join('path/to/your/images', row['filename'])  # Update this path
    image = load_img(img_path, target_size=image_size)
    image = img_to_array(image)
    images.append(image)
    labels.append(row['label'])

# Convert to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Preprocess images and labels
images = tf.keras.applications.xception.preprocess_input(images)
labels = tf.keras.utils.to_categorical(labels, num_classes=num_classes)

# Define the Xception model
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(images, labels, batch_size=batch_size, epochs=10, validation_split=0.2)

# Unfreeze some layers and fine-tune if necessary
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(images, labels, batch_size=batch_size, epochs=10, validation_split=0.2)