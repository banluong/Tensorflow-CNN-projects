# Script to implement AlexNet in TensorFlow 2.0
# Original paper:
# https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
# Original Authors and creators:
# Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Flatten, Dense, Dropout

IMG_HEIGHT = 227
IMG_WIDTH = 227

# Create model with Sequential
model = Sequential()

# First/Input layer
model.add(Conv2D(96, 11, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)))
model.add(MaxPooling2D())

# Second layer
model.add(Conv2D(256, 5, padding='same', activation='relu'))
model.add(MaxPooling2D())

# Third layer
model.add(ZeroPadding2D())
model.add(Conv2D(384, 3, padding='same', activation='relu'))

# Forth layer
model.add(ZeroPadding2D())
model.add(Conv2D(384, 3, padding='same', activation='relu'))

# Fifth layer
model.add(Conv2D(256, 3, padding='same', activation='relu'))
model.add(MaxPooling2D())

# Sixth layer
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Seventh layer
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))

# Final layer
model.add(Dense(1, activation='softmax'))
