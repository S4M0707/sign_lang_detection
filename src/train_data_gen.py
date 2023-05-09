#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:24:49 2023

@author: s4m0707
"""

import numpy as np
import os
import cv2

from mediapipe_processing import Mpipe_processing

# training data dir:
data_path = os.path.join(os.getcwd(), 'data')

actions = np.array(['hello', 'thanks', 'namaste'])

num_sequence = 30
len_sequence = 30

def make_data_dir():
    for action in actions:
        for sequence in range(num_sequence):
            try:
                os.makedirs(os.path.join(data_path, str(action), str(sequence)))
            except:
                pass

# training data video capturing
make_data_dir()
mp_pre = Mpipe_processing()

capture = cv2.VideoCapture(0)

with mp_pre.mpipe_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as holistic:
    
    for action in actions:
        for sequence in range(num_sequence):
            for frame_idx in range(len_sequence):
                
                # frame capture
                ret, frame = capture.read()

                result, img = mp_pre.mpipe_detection(frame, holistic)
                mp_pre.draw_landmarks(img, result)
                
                if frame_idx == 0:
                    cv2.putText(img, str(action), (120, 200),
                                cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4, cv2.LINE_AA)
                
                cv2.putText(img, 'Action: {} Sequence {}'.format(action, sequence), (15, 12),
                            cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                # exporting keypoints
                np_data = mp_pre.to_array_keypoints(result)
                np_path = os.path.join(data_path, str(action), str(sequence), str(frame_idx))
                np.save(np_path, np_data)

                cv2.imshow("Video", img)
                
                if frame_idx == 0:
                    cv2.waitKey(2000)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    capture.release()
    cv2.destroyAllWindows()