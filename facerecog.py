
from tensorflow.keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing import image
import cv2
import numpy as np
import tkinter as tk
from tkinter.ttk import *
import os
import random
from playsound import playsound

face_classifier = cv2.CascadeClassifier(r'/home/san/PycharmProjects/EmotionRecoginationbasedmusicplayer/haarcascade_frontalface_default.xml')
classifier = load_model(r'/home/san/PycharmProjects/EmotionRecoginationbasedmusicplayer/model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

cap =cv2.VideoCapture(0)

def playmusic(label):
    print(type(label))

    def plsong():
        if label == "Happy":
            music_dir = "/home/san/Music/happy"
            songs = os.listdir(music_dir)
            a = random.randint(0, len(songs) - 1)
            print(songs[a])
            cap.release()
            cv2.destroyAllWindows()
            playsound(f'{music_dir}/{songs[a]}')
            
        elif label == "Angry":
            music_dir = "/home/san/Music/angry"
            songs = os.listdir(music_dir)
            a = random.randint(0, len(songs) - 1)
            print(songs[a])
            cap.release()
            cv2.destroyAllWindows()
            playsound(f'{music_dir}/{songs[a]}')
            
        elif label == "Sad":
            music_dir = "/home/san/Music/sad"
            songs = os.listdir(music_dir)
            a = random.randint(0, len(songs) - 1)
            print(songs[a])
            cap.release()
            cv2.destroyAllWindows()
            playsound(f'{music_dir}/{songs[a]}')
        elif label=="Fear":
            music_dir = "/home/san/Music/fear"
            songs = os.listdir(music_dir)
            a = random.randint(0, len(songs) - 1)
            print(songs[a])
            cap.release()
            cv2.destroyAllWindows()
            playsound(f'{music_dir}/{songs[a]}')
            
        else:
            print("no songs")
            cap.release()
            cv2.destroyAllWindows()
    root = tk.Tk()
    frame = tk.Frame(root, width=100, height=100, background="bisque")
    frame.pack(fill='both', expand=True)

    Emotionlabel = tk.Label(frame,
                      text="You are feeling "+label).place(x=40, y=60)

    playm = tk.Button(frame,
                       text="playmusic",
                       command=plsong)
    playm.pack(side=tk.LEFT)

    root.mainloop()
while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)


        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)
            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            print(label)
        else:
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow('Emotion Detector',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        playmusic(label)
        break

cap.release()
cv2.destroyAllWindows()

