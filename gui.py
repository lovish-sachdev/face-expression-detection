import streamlit as st
import av
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer

def video_frame_callback(frame):
    success, image = video_frame_callback(frame)
    return av.VideoFrame.from_ndarray(frame, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
