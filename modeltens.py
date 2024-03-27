import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import av
import streamlit as st
from threading import Lock





from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard

# Initialize holistic model from MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Create the holistic model instance
holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load the model
@st.cache_resource()
def load_model():

    actions = np.array(['hello', 'thanks', 'i love you'])

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.load_weights('action.h5')
    return model



def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results




# Define function to extract keypoints
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])







# Define video frame callback function
sequence_lock = Lock()
sequence=[]
model = load_model()
predictions =[]
actions=np.array(['hello', 'thanks', 'iloveyou'])


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    
    image = frame.to_ndarray(format="bgr24")
    
    
    

    # Convert frame color to RGB

    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Process hand landmarks detection


    image,results = mediapipe_detection(rgb_image,holistic)
    
    

    # If hands are detected, draw landmarks and connections
    
    if results.face_landmarks:
        # print("face detected")
        overlay = image.copy()
        overlay[image.shape[0] // 100 :, :] = [0, 0, 0]
        alpha = 0.3 
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        res = extract_keypoints(results)
        with sequence_lock:
            if len(sequence)!=30:
                sequence.append(res)

            elif len(sequence) == 30:                
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                # predictions.append(np.argmax(res))


                cv2.putText(
                    image,
                    f"sign language prediction: {actions[np.argmax(res)]}",
                    (10, image.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                )

    return av.VideoFrame.from_ndarray(image, format="bgr24")

        

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)



