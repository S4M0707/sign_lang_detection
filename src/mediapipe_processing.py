#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:08:41 2023

@author: s4m0707
"""

import numpy as np
import mediapipe as mp
import cv2

class Mpipe_processing:
    
    def __init__(self):
        self.mpipe_drawing = mp.solutions.drawing_utils
        self.mpipe_holistic = mp.solutions.holistic
        
    def mpipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        result = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return result, image
    
    def draw_landmarks(self, image, result):
        
        # For face landmarks
        self.mpipe_drawing.draw_landmarks(image, result.face_landmarks, self.mpipe_holistic.FACEMESH_CONTOURS,
                                  self.mpipe_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                  self.mpipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        # For pose landmarks
        self.mpipe_drawing.draw_landmarks(image, result.pose_landmarks, self.mpipe_holistic.POSE_CONNECTIONS,
                                  self.mpipe_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1),
                                  self.mpipe_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1))
        
        # For left hand landmarks
        self.mpipe_drawing.draw_landmarks(image, result.left_hand_landmarks, self.mpipe_holistic.HAND_CONNECTIONS,
                                  self.mpipe_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=4),
                                  self.mpipe_drawing.DrawingSpec(color=(255, 255, 0), thickness=1, circle_radius=2))
        
        # For right hand landmarks
        self.mpipe_drawing.draw_landmarks(image, result.right_hand_landmarks, self.mpipe_holistic.HAND_CONNECTIONS,
                                  self.mpipe_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=4),
                                  self.mpipe_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=2))
    
    def to_array_keypoints(self, res_landmarks):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in res_landmarks.pose_landmarks.landmark]).flatten() if res_landmarks.pose_landmarks else np.zeros(33 * 4)
        left_hand = np.array([[res.x, res.y, res.z] for res in res_landmarks.left_hand_landmarks.landmark]).flatten() if res_landmarks.left_hand_landmarks else np.zeros(21 * 3)
        right_hand = np.array([[res.x, res.y, res.z] for res in res_landmarks.right_hand_landmarks.landmark]).flatten() if res_landmarks.right_hand_landmarks else np.zeros(21 * 3)
        face = np.array([[res.x, res.y, res.z] for res in res_landmarks.face_landmarks.landmark]).flatten() if res_landmarks.face_landmarks else np.zeros(468 * 3)
        
        return np.concatenate([face, pose, left_hand, right_hand])