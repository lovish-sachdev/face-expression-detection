import streamlit as st
import cv2
from keras.models import model_from_json
import numpy as np
import os

# Load model architecture from JSON file
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


def main():
    st.set_page_config(page_title="Streamlit WebCam App")
    st.title("Webcam Display Steamlit App")
    stop_button_pressed = st.button("Stop")
    start_button_pressed = st.button("start")
    haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_tracker=cv2.CascadeClassifier(haar_file)
    
    ## video processing
    if start_button_pressed:
        frame_placeholder = st.empty()
        cap = cv2.VideoCapture(-1)
        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_tracker.detectMultiScale(gray_frame,1.3,5)
            # st.write(tracked)
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

                #If one or more faces are found, draw a rectangle around the
                #largest face present in the picture
                if maxArea > 0 :
                    face_from_frame=gray_frame[y:y+h,x:x+w]
                    
                    resized=cv2.resize(face_from_frame,(48,48))
                                   
                    resized=np.expand_dims(resized,axis=0)
                                   
                    resized=np.expand_dims(resized,axis=3)
                    resized=resized.astype("float32")
                    resized=resized/255.

                    label=emotion_detection[np.argmax(loaded_model.predict(resized,verbose=0))]
                    cv2.putText(frame,label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
                    cv2.rectangle(frame,  (x-10, y-20),
	                		    (x + w+10 , y + h+20),
	            	    	    (0,255,0),2)
            frame_placeholder.image(frame,channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
        cap.release()
        cv2.destroyAllWindows()

    ###

if __name__ == "__main__":
    main()
