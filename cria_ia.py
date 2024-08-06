import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from tensorflow.keras.utils import to_categorical

# Carregar o dataset FER-2013
data = pd.read_csv('fer2013.csv')

# Preprocessamento dos dados
def preprocess_data(data):
    pixels = data['pixels'].tolist()
    width, height = 48, 48
    faces = []
    for pixel_sequence in pixels:
        face = np.array(pixel_sequence.split(), dtype='float32')
        face = face.reshape(width, height, 1)
        faces.append(face)
    faces = np.array(faces)
    faces = faces / 255.0
    emotions = to_categorical(data['emotion'], num_classes=7)
    return faces, emotions

faces, emotions = preprocess_data(data)

# Dividir os dados em treino e validação
num_samples, num_classes = emotions.shape
train_size = int(0.8 * num_samples)
train_faces, train_emotions = faces[:train_size], emotions[:train_size]
val_faces, val_emotions = faces[train_size:], emotions[train_size:]

# Construir o modelo
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Treinar o modelo
model.fit(train_faces, train_emotions, batch_size=64, epochs=30, validation_data=(val_faces, val_emotions), verbose=2)

# Salvar o modelo
model.save('emotion_model.h5')
