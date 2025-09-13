import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
import threading
from IPython.display import Audio, display



# Global flags
alarm_triggered = False

# Load model once
def load_model(path='./InceptionV3.h5'):
    try:
        model = tf.keras.models.load_model(path)
        print("✅ Model loaded successfully.")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Play alarm sound (non-blocking)
def play_alarm():
    try:
        display(Audio('./alarm.mp3', autoplay=True))
    except Exception as e:
        print(f"❌ Error playing alarm sound: {e}")

# Preprocess frame for prediction
def preprocess_frame(frame):
    im = Image.fromarray(frame, 'RGB')
    im = im.resize((224, 224))
    img_array = image.img_to_array(im)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Predict fire presence
def predict_fire(model, frame):
    processed = preprocess_frame(frame)
    probabilities = model.predict(processed)[0]
    prediction = np.argmax(probabilities)
    return prediction, probabilities

# Main loop for video processing
def process_video(video_path, model):
    global alarm_triggered
    video = cv2.VideoCapture(video_path)

    while True:
        grabbed, frame = video.read()
        if not grabbed:
            print("📭 No more frames to process.")
            break

        prediction, probabilities = predict_fire(model, frame)

        if prediction == 0:  # Fire detected
            print(f"🔥 Fire Detected! Probability: {probabilities[prediction]:.2f}")
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            if not alarm_triggered:
                threading.Thread(target=play_alarm).start()
                alarm_triggered = True

            cv2.imshow("Fire Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Exiting video loop.")
            break

    video.release()
    cv2.destroyAllWindows()

def detect_fire():
    
    video_loc = "C:\\Users\\Anurag\\Desktop\\Radhika_Project\\FireAlarmProject\\recordings\\recorded_video.mp4"
    model = load_model()
    if model:
        process_video(video_loc, model)

detect_fire()