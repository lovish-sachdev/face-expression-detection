import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
import os
from streamlit_webrtc import webrtc_streamer,RTCConfiguration
import av

class VideoProcessor():
    def recv(self,frame):
        frm=frame.to_ndarray(format="bgr24")
        gray_frame=cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
        faces=face_tracker.detectMultiScale(gray_frame,1.3,5)
        maxArea=0
        x = 0
        y = 0
        w = 0
        h = 0
        for (_x,_y,_w,_h) in faces:
            if  _w*_h > maxArea:
                x = _x
                y = _y
                w = _w
                h = _h
                maxArea = w*h
            if maxArea > 0 :
                face_from_frame=gray_frame[y:y+h,x:x+w]
                
                resized=cv2.resize(face_from_frame,(48,48))
                               
                resized=np.expand_dims(resized,axis=0)
                               
                resized=np.expand_dims(resized,axis=3)
                resized=resized.astype("float32")
                resized=resized/255.
                label=emotion_detection[np.argmax(loaded_model.predict(resized,verbose=0))]
                cv2.putText(frm,label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                cv2.rectangle(frm,  (x-10, y-20),
	            		    (x + w+10 , y + h+20),
	        	    	    (0,255,0),2)
        
        return av.VideoFrame.from_ndarray(frm,format="bgr24")


# # Load model architecture from JSON file
main_path=os.path.dirname(os.path.abspath(__file__))
json_file=os.path.join(main_path,"model.json")
model_path=os.path.join(main_path,"model15.h5")
with open(json_file, 'r') as json_file:
    loaded_model_json = json_file.read()

# Load model architecture
loaded_model = model_from_json(loaded_model_json)
emotion_detection = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')

# Load weights into the model
loaded_model.load_weights(model_path)

haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_tracker=cv2.CascadeClassifier(haar_file)
    

# vid_pro=VideoProcessor(loaded_model,face_tracker)
webrtc_streamer(key="key",video_processor_factory=VideoProcessor,rtc_configuration=RTCConfiguration({"iceServers":[{"urls":['stun:stun.l.google.com:19302']}]}))


