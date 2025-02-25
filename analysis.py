import streamlit as st
from parameters import *
import os
from ultralytics import YOLO
from ultralytics import RTDETR
import cv2
import torch
import supervision as sv
import numpy as np
from collections import defaultdict, deque
import plotly.express as px
import csv
import pandas as pd
from utils import create_heatmap, ViewTransformer, resize_frame_to_360p

st.set_page_config(page_title="Traffic Analysis", layout="wide")

st.title("Traffic Analysis")    

if os.path.exists(st.session_state["video_name"]):
    run = st.button("Run Application", type="secondary", use_container_width=True)
    message = st.empty()
    
    #region Dashboard setup
    with st.expander("Dashboard", expanded=True):
        st.info("This section displays the analysis results on the video frames.")
        section_dashboard_A = st.columns(2)   
        with section_dashboard_A[0]:
            sec_dash_A_1 = st.empty()
        with section_dashboard_A[1]:
            sec_dash_A_2 = st.empty()

        section_dashboard_B = st.columns(2)
        with section_dashboard_B[0]:
            sec_dash_B_1 = st.empty()
        with section_dashboard_B[1]:
            sec_dash_B_2 = st.empty()

        section_dashboard_C = st.columns(2)
        with section_dashboard_C[0]:
            sec_dash_C_1 = st.empty()
        with section_dashboard_C[1]:
            sec_dash_C_2 = st.empty()
    #endregion

    #region UI setup for the page
    with st.expander("Vehicle Detection", expanded=True):
        st.info("This section displays the detection results on the video frames.")
        num_detection_columns = len(st.session_state["annotators"])
        section_A = st.columns(num_detection_columns)
        detection_sections = []
        for i in range(num_detection_columns):
            detection_sections.append(section_A[i].empty())

    with st.expander("Vehicle Tracking", expanded=True):
        st.info("This section displays the tracking results on the video frames.")
        section_B = st.columns(2)
        with section_B[0]:
            flow_section_1 = st.empty()
        with section_B[1]:
            flow_section_2 = st.empty()

    with st.expander("Vehicle Flow Estimation", expanded=True):
        st.info("This section displays the flow estimation results on the video frames.")
        section_C = st.columns(2)
        with section_C[0]:
            count_section_1 = st.empty()
        with section_C[1]:
            count_section_2 = st.empty()

    with st.expander("Vehicle Density Analysis", expanded=True):
        st.info("This section displays the density analysis results on the video frames.")
        section_D = st.columns(2)
        with section_D[0]:
            sec_D_1 = st.empty()
        with section_D[1]:
            sec_D_2 = st.empty()

    with st.expander("Vehicle Speed Analysis", expanded=True):
        st.info("This section displays the speed analysis results on the video frames.")
        section_E = st.columns(2)
        with section_E[0]:
            sec_E_1 = st.empty()
        with section_E[1]:
            sec_E_2 = st.empty()

    #endregion

    if run:
        #region get the settings and set up the backend

        # get the detection settings
        confidence_threshold = st.session_state["confidence"]
        iou_threshold = st.session_state["iou"]
        det_mode = st.session_state["detection_mode"]
        selected_device = st.session_state["device"]
        selected_classes = st.session_state["selected_classes"]
        
        # get the model and class map
        if(st.session_state["model_map"][det_mode][0] == "detr"):
            model = RTDETR(st.session_state["model_map"][det_mode][1])
        else:
            model = YOLO(st.session_state["model_map"][det_mode][1])

        # set the device
        if selected_device == "GPU":
            if torch.cuda.is_available():
                message.success("GPU available. Switching to GPU.")
                device = "cuda"
            else:
                message.error("GPU not available. Switching to CPU.")
                device = "cpu"
        else:device = "cpu"   

        # get the class IDs to detect
        classID = []
        for cls in selected_classes:
            classID.append(st.session_state["class_map"][cls])   
        id_cls_map = {v: k for k, v in st.session_state["class_map"].items()}

        # initialize the tracker
        tracker = sv.ByteTrack()
        tracker.reset()

        # set up line zones
        line_params = st.session_state["line_params"]
        line_zones = []
        line_annotators = []
        for line in line_params:
            start_pt = sv.Point(line[0], line[1])
            end_pt = sv.Point(line[2], line[3])
            line_zones.append(sv.LineZone(start_pt, end_pt, triggering_anchors=[sv.Position.CENTER])) 
        line_annotators = []
        for i in range(len(line_zones)):
            line_annotators.append(sv.LineZoneAnnotator(color=sv.Color(161, 255, 70), text_scale=0.8, custom_in_text=f"Lane {i+1} [In]", 
                                                        custom_out_text=f"Lane {i+1} [Out]", text_thickness=1))
        
        # set up for heatmap
        heatmapAnnotator = SUPERVISION_ANNOTATORS["Heatmap"]
        track_history = defaultdict(lambda: [])
        last_positions = {}
        _im = st.session_state["frame0"]
        heatmap = np.zeros((_im.shape[0], _im.shape[1], 3), dtype=np.float32)

         # setup for speed analysis
        coordinates_for_speed = defaultdict(lambda: deque(maxlen=FRAME_RATE))
        x1, y1, x2, y2, x3, y3, x4, y4, bredth, length = st.session_state["zone_params"][0]
        width_buffer = st.session_state["speed_width_buffer"]
        height_buffer = st.session_state["speed_height_buffer"]
        SOURCE = np.array([
            [x1-width_buffer, y1-height_buffer],
            [x2-width_buffer, y2-height_buffer],
            [x4-width_buffer, y4-height_buffer],
            [x3-width_buffer, y3-height_buffer]
        ])
        TARGET_WIDTH = bredth
        TARGET_HEIGHT = length

        TARGET = np.array([
            [0, 0],
            [TARGET_WIDTH - 1, 0],
            [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
            [0, TARGET_HEIGHT - 1],
        ])
        view_transformer = ViewTransformer(source=SOURCE, target=TARGET)
        bounding_box_annotator = SUPERVISION_ANNOTATORS["Boxes"]
        
        #endregion

        #region data setup for visualization
        vehicle_distribution_map = {}
        vehicle_in_scene_map = {}
        lane_wise_vehicle_distribution_maps = []
        for i in range(len(line_zones)):
            lane_wise_vehicle_distribution_maps.append({})
        lane_wise_vehicle_in_scene_maps = []
        for i in range(len(line_zones)):
            lane_wise_vehicle_in_scene_maps.append({})        
        vehicle_speed_class_map = {}
        vehicle_speed_time_map = {}
        #endregion

        # loop through the frames
        capture = cv2.VideoCapture(st.session_state["video_name"])
        image_index = 0
        while capture.isOpened():
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            image_index+=1 # increment the image index
            image = resize_frame_to_360p(frame)
            
            #region dashboard update for each frame

            # vehicle distribution total
            types_list = vehicle_distribution_map.keys()
            count_list = vehicle_distribution_map.values()
            df = pd.DataFrame(list(zip(types_list, count_list)), columns=["Type", "Count"])
            fig = px.pie(df, values='Count', names='Type', title='Vehicle Distribution', hole=0.3)
            sec_dash_A_1.plotly_chart(fig)

            # vehicle distribution in lanes
            types_list_combined = []
            count_list_combined = []
            lane_list = []
            for i in range(len(lane_wise_vehicle_distribution_maps)):
                types_list_combined += lane_wise_vehicle_distribution_maps[i].keys()
                count_list_combined += lane_wise_vehicle_distribution_maps[i].values()
                lane_list += [f"Lane {i+1}"]*len(lane_wise_vehicle_distribution_maps[i])
            df = pd.DataFrame(list(zip(types_list_combined, count_list_combined, lane_list)), columns=["Type", "Count", "Lane"])
            fig = px.histogram(df, x='Type', y='Count', color='Lane', title='Vehicle Distribution by Lanes', labels={'Type': 'Type', 'Count': 'Count'}, barmode='group')
            sec_dash_A_2.plotly_chart(fig)

            # vehicle in scene total
            time_list = vehicle_in_scene_map.keys()
            count_list = vehicle_in_scene_map.values()
            df = pd.DataFrame(list(zip(time_list, count_list)), columns=["Time", "Count"])
            fig = px.line(df, x='Time', y='Count', title='Vehicle In Scene', markers=True, labels={'Time': 'Time (s)', 'Count': 'Count'})
            sec_dash_B_1.plotly_chart(fig)

            # vehicle in scene in lanes
            time_list_combined = []
            count_list_combined = []
            lane_list = []
            for i in range(len(lane_wise_vehicle_in_scene_maps)):
                time_list_combined += lane_wise_vehicle_in_scene_maps[i].keys()
                count_list_combined += lane_wise_vehicle_in_scene_maps[i].values()
                lane_list += [f"Lane {i+1}"]*len(lane_wise_vehicle_in_scene_maps[i])
            df = pd.DataFrame(list(zip(time_list_combined, count_list_combined, lane_list)), columns=["Time", "Count", "Lane"])
            fig = px.funnel(df, x='Time', y='Count', color='Lane', title='Vehicle In Scene by Lanes', labels={'Time': 'Time (s)', 'Count': 'Count'})
            sec_dash_B_2.plotly_chart(fig)

            # vehicle speed class wise
            types_list = vehicle_speed_class_map.keys()
            speed_list = [vehicle_speed_class_map[cls][0]/vehicle_speed_class_map[cls][1] for cls in types_list]
            df = pd.DataFrame(list(zip(types_list, speed_list)), columns=["Type", "Avg. Speed"])
            fig = px.bar_polar(df, r='Avg. Speed', theta='Type', title='Vehicle Speed Analysis', labels={'Type': 'Type', 'Speed': 'Speed (km/h)'}, color='Type')    
            sec_dash_C_1.plotly_chart(fig)

            # vehicle speed time wise
            time_list = vehicle_speed_time_map.keys()
            speed_list = vehicle_speed_time_map.values()
            df = pd.DataFrame(list(zip(time_list, speed_list)), columns=["Time", "Avg. Speed"])
            fig = px.area(df, x='Time', y='Avg. Speed', title='Vehicle Speed Analysis', labels={'Time': 'Time (s)', 'Avg. Speed': 'Avg. Speed (km/h)'})
            sec_dash_C_2.plotly_chart(fig)

            #endregion

            # perform detection
            _image = image.copy()
            results = model.track(_image, conf=confidence_threshold, iou=iou_threshold, device=device, classes=classID, max_det=1000)

            # get the detections
            detections = sv.Detections.from_ultralytics(results[0])
            # update the tracker
            detections = tracker.update_with_detections(detections)

            #region show the detections
            for i in range(num_detection_columns):
                # annotate the frame
                annotated_frame = SUPERVISION_ANNOTATORS[st.session_state["annotators"][i]].annotate(_image.copy(), detections=detections)
                # display the annotated frame
                detection_sections[i].image(annotated_frame, channels="BGR", use_column_width=True)
            #endregion

            #region show tracking
            # display the flow on original image
            annotated_frame = SUPERVISION_ANNOTATORS["Dot"].annotate(_image.copy(), detections=detections)
            annotated_frame = SUPERVISION_ANNOTATORS["Trace"].annotate(annotated_frame, detections=detections)
            flow_section_1.image(annotated_frame, channels="BGR", use_column_width=True)
            
            # display the flow on dark background
            overlay = _image.copy()
            output = _image.copy()
            alpha = 0.9
            cv2.rectangle(overlay, (0, 0), (_image.shape[1], _image.shape[0]), (0, 0, 0), -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            annotated_frame = SUPERVISION_ANNOTATORS["Dot"].annotate(output, detections=detections)
            annotated_frame = SUPERVISION_ANNOTATORS["Trace"].annotate(annotated_frame, detections=detections)
            flow_section_2.image(annotated_frame, channels="BGR", use_column_width=True)
            #endregion

            #region show counting and flow estimation
            # vechicle counting
            label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=5, text_position=sv.Position.TOP_CENTER)
            
            _labels = [f"{id_cls_map[cls].upper()}" for cls in detections.class_id]
            annotated_frame = label_annotator.annotate(_image.copy(), detections=detections, labels=_labels)
            count_section_1.image(annotated_frame, channels="BGR", use_column_width=True)
            sec_E_1.image(annotated_frame, channels="BGR", use_column_width=True)

            # line zone detection
            annotated_frame = _image.copy()
            for line_zone, line_annotator in zip(line_zones, line_annotators):
                trigger = line_zone.trigger(detections=detections)
                annotated_frame = line_annotator.annotate(annotated_frame, line_counter=line_zone)
                
                for isIn, isOut, cls in zip(trigger[0], trigger[1], detections.class_id):
                    if isIn or isOut:
                        v_class = id_cls_map[cls].upper()
                        if v_class in vehicle_distribution_map:vehicle_distribution_map[v_class] += 1
                        else:vehicle_distribution_map[v_class] = 1

                        if v_class in lane_wise_vehicle_distribution_maps[line_zones.index(line_zone)]:
                            lane_wise_vehicle_distribution_maps[line_zones.index(line_zone)][v_class] += 1
                        else:
                            lane_wise_vehicle_distribution_maps[line_zones.index(line_zone)][v_class] = 1

                if image_index%FRAME_RATE == 0:
                    in_out_count = line_zone.in_count + line_zone.out_count
                    prev_frame = image_index//FRAME_RATE - 1
                    if prev_frame in lane_wise_vehicle_in_scene_maps[line_zones.index(line_zone)]:
                        lane_wise_vehicle_in_scene_maps[line_zones.index(line_zone)][image_index//FRAME_RATE] = in_out_count - lane_wise_vehicle_in_scene_maps[line_zones.index(line_zone)][prev_frame]
                    else:
                        lane_wise_vehicle_in_scene_maps[line_zones.index(line_zone)][image_index//FRAME_RATE] = in_out_count
                        

            count_section_2.image(annotated_frame, channels="BGR", use_column_width=True)
            
            if image_index%FRAME_RATE == 0:
                vehicle_in_scene_map[image_index//FRAME_RATE] = len(detections.class_id)        
            #endregion

            #region show density analysis
            # heatmap 1
            annotated_frame = heatmapAnnotator.annotate(_image.copy(), detections=detections)
            sec_D_1.image(annotated_frame, channels="BGR", use_column_width=True)   

            # heatmap 2
            trck_ids = [id for id in detections.tracker_id]   
            overlay, heatmap, track_history, last_positions = create_heatmap(results, trck_ids, track_history, last_positions, _image.copy(), heatmap)
            sec_D_2.image(overlay, channels="BGR", use_column_width=True)
            #endregion             

            #region show speed analysis
            
            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coordinates_for_speed[tracker_id].append(y)
            # format labels
            labels = []

            instance_map = {}
            total_speed = 0
            total_count = 0

            for tracker_id, class_id in zip(detections.tracker_id, detections.class_id):
                if len(coordinates_for_speed[tracker_id]) < FRAME_RATE / 2:
                    labels.append(f"Analysing...")
                else:
                    # calculate speed
                    coordinate_start = coordinates_for_speed[tracker_id][-1]
                    coordinate_end = coordinates_for_speed[tracker_id][0]
                    distance = abs(coordinate_start - coordinate_end)
                    time = len(coordinates_for_speed[tracker_id]) / FRAME_RATE
                    speed = distance / time * 3.6
                    # labels.append(f"#{tracker_id}")
                    labels.append(f"{int(speed)} km/h")
                    
                    cls_ = id_cls_map[class_id].upper()
                    if cls_ not in vehicle_speed_class_map:
                        vehicle_speed_class_map[cls_] = [speed, 1]
                    else:
                        vehicle_speed_class_map[cls_][0] += speed
                        vehicle_speed_class_map[cls_][1] += 1

                    total_speed += speed
                    total_count += 1
            
            if image_index%FRAME_RATE == 0:
                if total_count > 0:
                    vehicle_speed_time_map[image_index//FRAME_RATE] = total_speed/total_count
                else:
                    vehicle_speed_time_map[image_index//FRAME_RATE] = 0

            annotated_frame = bounding_box_annotator.annotate(
                scene=image.copy(), detections=detections
            )
            annotated_frame = label_annotator.annotate(
                scene=annotated_frame, detections=detections, labels=labels
            )
            sec_E_2.image(annotated_frame, channels="BGR", use_column_width=True)

            #endregion
        
        # release the video capture
        capture.release()
else:
    # display error message if no video file is uploaded
    st.error("Please upload a video file first.")