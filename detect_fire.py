import os
import streamlit as st
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
from IPython.display import Audio, display
import threading
import time
detection = 0
# --- Configuration ---
# Set this to True if you are running in a Google Colab notebook
IN_COLAB = False 
ALARM_SOUND_PATH = 'alarm.mp3'
VIDEO_FILE_PATH = 'recordings/recorded_video.mp4'
MODEL_PATH = 'InceptionV3.h5' 
INPUT_IMAGE_SIZE = (224, 224) # Correct size based on your error message

# --- Global Variables ---
Alarm_Status = False

# --- Function to play alarm sound ---
def play_alarm_sound_function():
    global Alarm_Status
    if IN_COLAB:
        print("Alarm triggered! (Colab Audio might not play)")
    else:
        try:
            from playsound import playsound
            playsound(ALARM_SOUND_PATH, True)
        except ImportError:
            print("playsound library not found. Please install with 'pip install playsound'.")
        except Exception as e:
            print(f"Error playing sound: {e}")
    time.sleep(10) # Play for 10 seconds to avoid repeating
    Alarm_Status = False

# --- Main Video Processing Logic ---
def run_detection():
    global Alarm_Status
    global detection
    
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    video = cv2.VideoCapture(VIDEO_FILE_PATH)
    if not video.isOpened():
        print(f"Error: Could not open video file at {VIDEO_FILE_PATH}")
        return

    while True:
        grabbed, frame = video.read()
        if not grabbed:
            print('End of video or no frame available.')
            break

        # Resize the frame to match the input size of your trained model (224x224)
        frame_resized = cv2.resize(frame, INPUT_IMAGE_SIZE)
        
        # Prepare the image for the model
        img_array = keras_image.img_to_array(frame_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  # Rescale to [0,1]
        
        # Make prediction
        # The `verbose=0` parameter is a good practice for real-time applications
        probabilities = model.predict(img_array, verbose=0)[0]
        prediction = np.argmax(probabilities)
        
        # Check for fire (assuming class 0 is 'fire')
        if prediction == 0:
            print(f"Fire Detected: Estimated Probability {probabilities[prediction]:.2f}")
            detection = detection + 1
            if not Alarm_Status:
                # threading.Thread(target=play_alarm_sound_function, daemon=True).start()
                play_alarm_sound_function()
                Alarm_Status = True
            
            # Optionally add a red overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), -1)
            alpha = 0.4
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            cv2.imshow('Fire Detection', frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st.error("Red Flag: Fire is detected. Be cautious!!")
            st.image(frame_rgb, caption="Fire Detected", use_container_width=True)
            
        # Display the frame
        # if IN_COLAB:
        #     from matplotlib import pyplot as plt
        #     plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        #     plt.show()
        # else:
        #     cv2.imshow('Fire Detection', frame)

  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if detection > 5:
            st.error("Red Flag: Fire is detected. Be cautious!!")
            st.image(frame_rgb, caption="Fire Detected", use_container_width=True)
            return("yes")
            
            


    video.release()
    cv2.destroyAllWindows()
    


    

# run_detection()
