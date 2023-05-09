#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 21:31:45 2023

@author: s4m0707
"""

import os
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
from keras.layers import LSTM, Dense

class Model:
    def __init__(self):
        self.data_path = os.path.join(os.getcwd(), os.pardir, 'data')
        
        # self.actions = np.array([action for action in os.listdir(self.data_path)])
        self.actions = np.array(['hello', 'thanks', 'namaste'])
        print(self.actions)
        
        self.num_sequence = len(os.listdir(os.path.join(self.data_path, str(self.actions[0]))))
        print(self.num_sequence)
        
        self.len_sequence = len(os.listdir(os.path.join(self.data_path, str(self.actions[0]), str('0'))))
        print(self.len_sequence)
        
        # self.class_map = {key:value for value, key in enumerate(os.listdir(self.data_path))}
        self.class_map = {key:value for value, key in enumerate(self.actions)}
        print(self.class_map)
        
        features = []
        classes = []
        for action in self.actions:
            for sequence in range(self.num_sequence):
                window=[]
                
                for frame_idx in range (self.len_sequence):
                    res = np.load(os.path.join(self.data_path, str(action), str(sequence), "{}.npy".format(frame_idx)))
                    window.append(res)
                    
                features.append(window)
                classes.append(self.class_map[action])
        
        X = np.array(features)
        y = keras.utils.to_categorical(classes).astype(int)
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.05)
        
        print(self.X_train.shape)
        print(self.y_train.shape[1])
        
        log_path = os.path.join(os.getcwd(), os.pardir, 'logs')
        
        try:
            os.mkdir(log_path)
        except:
            pass
        
        self.tb_callbacks = keras.callbacks.TensorBoard(log_dir=log_path)
        
        self.model = keras.models.Sequential()
        self.model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(30, 1662)))
        # self.model.add(LSTM(32, return_sequences=True, activation='relu'))
        self.model.add(LSTM(32, return_sequences=False, activation='relu'))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.actions.shape[0], activation='softmax'))
        
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    def train_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=100, callbacks=[self.tb_callbacks])
        save_path = os.path.join(os.getcwd(), os.pardir, 'action.h5')
        self.model.save(save_path)
        
        test_res = self.model.predict(self.X_test)
        for i in range(len(self.X_test)):
            print(self.actions[np.argmax(test_res[i])], self.actions[np.argmax(self.y_test[i])])
            
        return self.model
    
    def pretrained_model(self):
        save_path = os.path.join(os.getcwd(), os.pardir, 'action.h5')
        self.model.load_weights(save_path)
        
        test_res = self.model.predict(self.X_test)
        for i in range(len(self.X_test)):
            print(self.actions[np.argmax(test_res[i])], self.actions[np.argmax(self.y_test[i])])
            
        return self.model
    
obj = Model()
x = obj.pretrained_model()