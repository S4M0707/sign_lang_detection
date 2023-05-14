# sign_lang_detection

This project is built to help read and interprete sign languages of India. We are using an LSTM & DENSE model as base.

## Technologies used

- python 3.9
- cv2
- mediapipe
- numpy

## To run

- Codes are in the src folder.
- to run the project run the **output.py** file in src or simply the **run.sh** in the base directory

## python files

All the python codes in the src directory are used for:
- **mediapipe_processing.py** : operations of mediapipe module on a perticular frame and generating final numpy file for model
- **train_data_gen.py** : to capture video data for every action such as 'thanks', 'hello', etc. and saving them in data directory
- **model_base.py** : this file contains the LSTM, DENSE model which is in form of a class with 2 member functions *train_model()* & *pretrained_model()*.
- **output.py** : this is the final file which shows the output that is the sign language detection
[NOTE: to use pretrained_model() there must be the action.h5 file present in base directory]


## Demo

![Demo](./demo.gif)