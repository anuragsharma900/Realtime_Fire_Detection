import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import os
from datetime import datetime
import detect_fire as fire



# Ensure a folder for recordings
os.makedirs("recordings", exist_ok=True)

def generate_filename():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join("recordings", f"recorded_video.mp4")

class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.recording = False
        self.out = None
        self.filename = generate_filename()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        if self.recording:
            if self.out is None:
                height, width = img.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(self.filename, fourcc, 20.0, (width, height))
                print(f"Started recording to {self.filename}")
            self.out.write(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def stop_recording(self):
        if self.out:
            self.out.release()
            self.out = None
            print(f"Recording stopped. File saved: {self.filename}")
            return self.filename
        return None
        

def start_rec():
    st.set_page_config(page_title="Fire Detection Model", layout="centered")
    st.title("🎥 Fire Detection ~ CCTV Footage Recording")
    # Store recording state in session
    if "recording" not in st.session_state:
        st.session_state.recording = False

    # Create the webrtc streamer and get its context
    webrtc_ctx = webrtc_streamer(
        key="video",
        video_processor_factory=VideoRecorder,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

    # Control buttons
    col1, col2, col3 = st.columns(3)
    

    with col1:
        if st.button("▶️ Start Recording"):
            st.session_state.recording = True
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.recording = True
                st.success("Recording started...")
            else:
               st.error("Recording Again Or Check Camera")

    with col2:
        if st.button("⏹️ Stop Recording"):
            st.session_state.recording = False
            if webrtc_ctx.video_processor:
                webrtc_ctx.video_processor.recording = False
                saved_file = webrtc_ctx.video_processor.stop_recording()
                if saved_file and os.path.exists(saved_file):
                    st.success(f"Recording saved as `{saved_file}`")
                    
                else:
                    st.error("Recording failed or file not found.")
            else:
                st.error("Recording Again Or Check Camera")

    with col3:
        if st.button("🎥 Detect Fire from the footage"):
            with st.spinner("Processing... Please wait."):
                result = fire.run_detection()
                if result == "yes":
                    st.error("Red Flag: Fire is detected. Be cautious!!")
                else:
                    st.success("Green Flag: No Fire Detected")
                    
            # st.session_state.recording = False