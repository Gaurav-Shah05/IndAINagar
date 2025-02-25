import streamlit as st
from utils import resize_frame_to_360p
from parameters import *
import os
import cv2
import shutil
import random
import string

if os.path.exists(DATA_PATH):
    shutil.rmtree(DATA_PATH)
os.makedirs(DATA_PATH)

st.set_page_config(page_title="Upload", layout="wide")

st.title("Traffic Analysis using Computer Vision")

st.success("To get started, upload a video file")

uploaded_file = st.file_uploader("", type=["mp4", "mov", "avi", "asf", "wmv", "flv", "mkv", "webm"], 
                                    key="upload_video", help="Upload a video file", accept_multiple_files=False)

col = st.columns(3)
if uploaded_file is not None:

    st.session_state["video_name"] = ''.join(random.choices(string.ascii_uppercase + string.digits, k=30))
    st.session_state["video_name"] = os.path.join(DATA_PATH, st.session_state["video_name"] + ".mp4")

    with open(st.session_state["video_name"], "wb") as f:
        f.write(uploaded_file.getbuffer())

    # if os.path.exists(IMAGE_PATH):
    #     shutil.rmtree(IMAGE_PATH)
    # os.makedirs(IMAGE_PATH)
    # video2images(st.session_state["video_name"], image_path=IMAGE_PATH, size=VIDEO_SIZE, resize=False)

    cap = cv2.VideoCapture(st.session_state["video_name"])
    ret, frame = cap.read()
    frame = resize_frame_to_360p(frame)
    st.session_state["frame0"] = frame
    cap.release()
    
    st.switch_page("settings.py")