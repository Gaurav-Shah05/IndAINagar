import streamlit as st
from parameters import *
import os
from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics import FastSAM
import cv2
import torch
import supervision as sv
import numpy as np
import shutil

st.set_page_config(page_title="Detection", layout="wide")

st.title("Settings")

# check if the video file is uploaded
if os.path.exists(st.session_state["video_name"]):

    #region Models Settings
    with st.expander("Model Settings", expanded=True):
        st.info("Select the model set, mode, device, confidence threshold, and IoU threshold.")
        col_0 = st.columns(2)
        with col_0[0]:
            # conficence threshold    
            st.session_state["confidence"] = st.slider("Set the Confidence Threshold", 0.0, 1.0, 0.5, 0.01, help="Setting the confidence threshold determines the minimum probability required for a bounding box to be considered a valid detection, filtering out less certain predictions.")
        with col_0[1]:
            st.session_state["iou"] = st.slider("Set the IoU Threshold", 0.0, 1.0, 0.7, 0.01, help="Setting the IoU threshold determines the minimum overlap required between two bounding boxes to be considered the same object.")
        
        col_A = st.columns(3) 
        with col_A[0]:
            # select the model set
            selected_model_set = st.selectbox("Select the Model Set", list(MODEL_SETS.keys()), help="Select the model set based on the desired task and requirements.") 
        with col_A[1]:
            # select the model map based on the selected model set
            st.session_state["model_map"] = MODEL_SETS[selected_model_set][0]
            # select the detection mode
            st.session_state["detection_mode"] = st.selectbox("Select the Mode", list(st.session_state["model_map"].keys()), help="Select the detection mode based on the desired trade-off between speed and accuracy.")
        with col_A[2]:
            # select the device
            st.session_state["device"] = st.selectbox("Select the Device", ["CPU", "GPU"], help="Select the device to run the detection model on.")
    #endregion

    #region Enhancement Settings
    with st.expander("Enhancement Settings", expanded=True):
        st.info("Configure video enhancement for low-light conditions.")
        
        col_enh = st.columns(2)
        with col_enh[0]:
            # Enhancement mode selection
            enhancement_mode = st.selectbox(
                "Enhancement Mode",
                ENHANCEMENT_MODES,
                index=ENHANCEMENT_MODES.index(DEFAULT_ENHANCEMENT_MODE),
                help="Choose whether to enhance video, and how to decide when to enhance."
            )
            
            # Store the selection in session state
            st.session_state["enhancement_mode"] = enhancement_mode
        
        with col_enh[1]:
            # If Auto mode is selected, show threshold slider
            if enhancement_mode == "Auto":
                enhancement_threshold = st.slider(
                    "Brightness Threshold",
                    0, 100, DEFAULT_ENHANCEMENT_THRESHOLD, 1,
                    help="Frames with average brightness below this threshold will be enhanced."
                )
                st.session_state["enhancement_threshold"] = enhancement_threshold
            else:
                # Set a default threshold even if not shown
                st.session_state["enhancement_threshold"] = DEFAULT_ENHANCEMENT_THRESHOLD
        
        # Model selection for future extension
        if enhancement_mode != "Off":
            enhancement_model = st.selectbox(
                "Enhancement Model",
                list(ENHANCEMENT_MODELS.keys()),
                index=0,
                help="Select the enhancement model to use."
            )
            st.session_state["enhancement_model"] = enhancement_model
            st.session_state["enhancement_model_path"] = ENHANCEMENT_MODELS[enhancement_model]["weight_path"]
    #endregion

    # region Detector Settings
    with st.expander("Detector Settings", expanded=True):
        st.info("Select the annotators to display the detection results on the dashboard and the classes to detect in the video.")
        st.session_state["annotators"] = st.multiselect("Configure Detection Dashboard", ["Boxes", "Corners", "Color", "Circle", "Ellipse", "Percentage Bar"], ["Color", "Corners"], help="Select the annotators to display the detection results on the dashboard.")
        # select the class map based on the selected model set
        st.session_state["class_map"] = MODEL_SETS[selected_model_set][1]
        _classes = list(st.session_state["class_map"].keys())
        # select the classes to detect
        st.session_state["selected_classes"] = st.multiselect("Classes to Detect", _classes, _classes, help="Select the classes to detect in the video.")
    #endregion

    #region Counting Settings
    with st.expander("Flow Estimation Settings", expanded=True):
        st.info("Select the number of detection lines and configure the lines to count the objects crossing the line.")

        
        frame0 = st.session_state["frame0"]
        line_params = []

        col_B = st.columns(2)

        with col_B[0]:
            num_lines = st.number_input("Number of detection Lines", 1, 3, 1, 1, help="Select the number of lines to count the objects crossing the line.")
            line_info = []
            im = frame0.copy()
            im_width = im.shape[1]
            im_height = im.shape[0]
            for i in range(num_lines):
                line_info.append(st.columns(4))
            for i in range(num_lines):
                x1 = line_info[i][0].slider("X1", 0, im_width - int(0.1 * im_width), 0, 1, key=f"x1_{i}")
                y1 = line_info[i][1].slider("Y1", 0, im_height - int(0.1 * im_height), im_height//2, 1, key=f"y1_{i}")
                x2 = line_info[i][2].slider("X2", int(0.1 * im_width), im_width, im_width, 1, key=f"x2_{i}")
                y2 = line_info[i][3].slider("Y2", int(0.1 * im_height), im_height, im_height//2, 1, key=f"y2_{i}")
                line_params.append([x1, y1, x2, y2])
        with col_B[1]:
            im = frame0.copy()
            # draw the lines on the image
            for line in line_params:
                x1, y1, x2, y2 = line
                cv2.line(im, (x1, y1), (x2, y2), (161, 255, 170), 2)
            st.image(im, channels="BGR", use_column_width=True)

        st.session_state["line_params"] = line_params
    #endregion

    #region Speed Estimation Settings
    with st.expander("Speed Estimation Settings", expanded=True):
        st.info("Select the number of detection regions and configure the regions to estimate the speed of the objects.")

        frame0 = st.session_state["frame0"]
        zone_params = []

        col_C = st.columns(2)
        img_for_zone = frame0.copy()
        # add black padding to the image with width and height of 10% of the image size
        height_buffer = int(0.4 * img_for_zone.shape[0])
        width_buffer = int(0.4 * img_for_zone.shape[1])
        img_for_zone = cv2.copyMakeBorder(img_for_zone, height_buffer, height_buffer, width_buffer, width_buffer, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        with col_C[0]:
            # num_zones = st.number_input("Number of detection Lines", 1, 3, 1, 1, help="Select the number of zones to estimate the speed of the objects.")
            zone_info = []
            im = img_for_zone.copy()

            im_width = im.shape[1]
            im_height = im.shape[0]
            for i in range(1):
                zone_info.append(st.columns(4))
                zone_info.append(st.columns(2))
            for i in range(1):
                x1 = zone_info[i][0].slider("X Top Left", 0, im_width - int(0.1 * im_width), width_buffer, 1, key=f"x11_{i}")
                y1 = zone_info[i][1].slider("Y Top Left", 0, im_height - int(0.1 * im_height), height_buffer, 1, key=f"y11_{i}")
                x2 = zone_info[i][2].slider("X Top Right", int(0.1 * im_width), im_width, im_width-width_buffer, 1, key=f"x12_{i}")
                y2 = zone_info[i][3].slider("Y Top Right", int(0.1 * im_height), im_height, height_buffer, 1, key=f"y12_{i}")
                x3 = zone_info[i][0].slider("X Bottom Left", 0, im_width - int(0.1 * im_width), width_buffer, 1, key=f"x21_{i}")
                y3 = zone_info[i][1].slider("Y Bottom Left", 0, im_height - int(0.1 * im_height), im_height-height_buffer, 1, key=f"y21_{i}")
                x4 = zone_info[i][2].slider("X Bottom Right", int(0.1 * im_width), im_width, im_width-width_buffer, 1, key=f"x22_{i}")
                y4 = zone_info[i][3].slider("Y Bottom Right", int(0.1 * im_height), im_height, im_height-height_buffer, 1, key=f"y22_{i}")
                bredth = zone_info[i+1][0].slider("Bredth of Road in Meters", 1, 200, 20, 1, key=f"bredth_{i}")
                length = zone_info[i+1][1].slider("Length of Road in Meters", 1, 500, 100, 1, key=f"length_{i}")
                zone_params.append([x1, y1, x2, y2, x3, y3, x4, y4, bredth, length])
        
        with col_C[1]:
            im = img_for_zone.copy()
            _colors = [(108, 66, 245), (245, 121, 69), (142, 245, 51)]
            # draw the zones on the image
            for zone_idx, zone in enumerate(zone_params):
                x1, y1, x2, y2, x3, y3, x4, y4 = zone[:8]
                cv2.line(im, (x1, y1), (x2, y2), _colors[zone_idx%len(_colors)], 3)
                cv2.line(im, (x2, y2), (x4, y4), _colors[zone_idx%len(_colors)], 3)
                cv2.line(im, (x4, y4), (x3, y3), _colors[zone_idx%len(_colors)], 3)
                cv2.line(im, (x3, y3), (x1, y1), _colors[zone_idx%len(_colors)], 3)                
            st.image(im, channels="BGR", use_column_width=True)

        st.session_state["zone_params"] = zone_params
        st.session_state["speed_width_buffer"] = width_buffer
        st.session_state["speed_height_buffer"] = height_buffer
    #endregion

    # #region Dashboard Settings
    # st.info("Dashboard Settings")
    # st.info("Select the components to display on the dashboard.")   
    # col_C = st.columns(4)
    # with col_C[0]:
    #     st.session_state["show_detection"] = st.checkbox("Show Detection", value=True)
    # with col_C[1]:
    #     st.session_state["show_tracking"] = st.checkbox("Show Tracking", value=True)
    # with col_C[2]:
    #     st.session_state["show_counting"] = st.checkbox("Show Counting", value=True)
    # with col_C[3]:
    #     st.session_state["show_density"] = st.checkbox("Show Density", value=True)
    # st.write("---")
    # #endregion

    run = st.button("Go to Analysis", type="primary", use_container_width=True, help="Click to switch to the analysis page.")

    if run:        
        # switch to the analysis page
        st.switch_page("analysis.py")  
else:
    # if the video file is not uploaded
    st.error("Please upload a video file first.")