# Import necessary libraries
import mediapipe as mp
import cv2
import numpy as np
import uuid
import os
import av

import streamlit as st

from streamlit_webrtc import webrtc_streamer,VideoTransformerBase,WebRtcMode



mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Initialize the hands model
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)



def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")

    # Convert frame color to RGB

    
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


    # Process hand landmarks detection


    results = hands.process(rgb_image)


    # Add a dark overlay to the bottom of the frame

    

     # Set the bottom part to black

     # Adjust the alpha value for transparency
    

    # If hands are detected, draw landmarks and connections
    if results.multi_hand_landmarks:
        overlay = image.copy()
        overlay[image.shape[0] // 100 :, :] = [0, 0, 0] 
        alpha = 0.3 
        image = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

        # Add text annotation at the bottom
        cv2.putText(
            image,
            "Hand detected",
            (10, image.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
        )

    return av.VideoFrame.from_ndarray(image, format="bgr24")





webrtc_streamer(key="example", 
                mode=WebRtcMode.SENDRECV,
                media_stream_constraints={"video": True, "audio": False},
                video_frame_callback=video_frame_callback,
                async_processing=True)











