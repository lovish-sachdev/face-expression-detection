import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # You can modify the frame here
        return frame

def main():
    st.title("Webcam Viewer with Streamlit and WebRTC")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_transformer_factory=VideoTransformer,
        client_settings=ClientSettings(
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        async_transform=True
    )

    if webrtc_ctx.video_transformer:
        st.write("Press 'q' to quit the webcam.")

        # Check if the user pressed 'q' to quit
        if st.button("Quit (Press 'q')"):
            webrtc_ctx.stop()

if __name__ == "__main__":
    main()
