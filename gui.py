import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        return frame

def main():
    st.title("Webcam Viewer with Streamlit and WebRTC")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    if webrtc_ctx.video_transformer:
        st.write("Press 'q' to quit the webcam.")

        # Check if the user pressed 'q' to quit
        if st.button("Quit (Press 'q')"):
            webrtc_ctx.stop()

if __name__ == "__main__":
    main()
