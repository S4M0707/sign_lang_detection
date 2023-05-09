#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 23:35:21 2023

@author: s4m0707
"""

import numpy as np
import cv2

from model_base import Model
from mediapipe_processing import Mpipe_processing

model_obj = Model()
model = model_obj.pretrained_model()

actions = model_obj.actions;

def visualize_prediction(pred, image):
    img = image.copy()
    colors = [(202, 180, 255), (255, 255, 0), (168, 168, 168)]
    for num, prob in enumerate(pred):
        cv2.rectangle(img, (0, 100+num*100), (int(prob * 200), 150+num*100), colors[num], -1)
        cv2.putText(img, actions[num], (8, 140+num*100),
                    cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 0, 255), 2, cv2.LINE_AA)
        
    return img

mp_pre = Mpipe_processing()

sequence = []
capture = cv2.VideoCapture(0)

with mp_pre.mpipe_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    while capture.isOpened():
        
        ret, frame = capture.read()
        
        result, img = mp_pre.mpipe_detection(frame, holistic)
        mp_pre.draw_landmarks(img, result)
        
        np_data = mp_pre.to_array_keypoints(result)
        sequence.append(np_data)
        
        if len(sequence) > 30:
            sequence = sequence[1:]
        
        if len(sequence) == 30:
            pred = model.predict(np.expand_dims(sequence, axis=0))[0]
            img = visualize_prediction(pred, img)

        cv2.imshow("Video", img)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()