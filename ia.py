import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Carregar o modelo pré-treinado
model = load_model('emotion_model.h5')

# Lista de emoções que o modelo pode reconhecer
emotion_labels = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']

def load_model_and_labels(model_path):
    model = load_model(model_path)
    emotion_labels = ['Raiva', 'Nojo', 'Medo', 'Feliz', 'Triste', 'Surpreso', 'Neutro']
    return model, emotion_labels

def preprocess_image(face_image):
    roi_gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
    if np.sum([roi_gray]) != 0:
        roi = roi_gray.astype('float') / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        return roi
    else:
        return None

def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

def predict_emotion(face_image, model, emotion_labels):
    roi = preprocess_image(face_image)
    if roi is not None:
        preds = model.predict(roi)[0]
        label = emotion_labels[preds.argmax()]
        return label
    else:
        return None

def draw_label(frame, faces, emotion_labels):
    for (x, y, w, h) in faces:
        face_image = frame[y:y+h, x:x+w]
        emotion = predict_emotion(face_image, model, emotion_labels)
        if emotion is not None:
            label_position = (x, y)
            cv2.putText(frame, emotion, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    return frame

def main():
    model_path = 'emotion_model.h5'
    model, emotion_labels = load_model_and_labels(model_path)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        faces = detect_faces(frame)
        frame = draw_label(frame, faces, emotion_labels)
        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
