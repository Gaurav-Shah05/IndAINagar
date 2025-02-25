import streamlit as st
import supervision as sv
from ultralytics import YOLO
from ultralytics import RTDETR

PAGES = {
    # "Upload a video file": [
    #     st.Page("upload.py", title="Upload a video file"),
    # ],
    # "Anatysis": [
    #     st.Page("settings.py", title="Settings"),
    #     st.Page("analysis.py", title="Analysis"),
    # ],
    "": [
        st.Page("upload.py", title="Upload Video"),
        st.Page("settings.py", title="Dashbaord Settings"),
        st.Page("analysis.py", title="Analysis Dashboard"),
    ],
}

VIDEO_SIZE = (640, 640)
DATA_PATH = "data/"
VIDEO_NAME = DATA_PATH + "_input_.mp4"
IMAGE_PATH = DATA_PATH + "frames/"
FRAME_RATE = 25

MODEL_MAP_SET_A = {
    "Light (Faster)": ("yolo", "dependencies/set_a/yolov10n-best.pt"),
    "Medium (Balanced)": ("detr", "dependencies/set_a/rtdetr-l-best.pt"),
    "Heavy (Better Accuracy)": ("yolo", "dependencies/set_a/yolov10x-best.pt"),
}

SET_A_CLS_MAP = {
  'Ambulance':0,
  'Army Vehicle':1,
  'Auto':2,
  'Bicycle':3,
  'Bus':4,
  'Car':5,
  'Garbagevan':6,
  'Human Hauler':7,
  'Minibus':8,
  'Minivan':9,
  'Motorbike':10,
  'Pickup':11,
  'Policecar':12,
  'Rickshaw':13,
  'Scooter':14,
  'SUV':15,
  'Taxi':16,
  'CNG Three Wheeler':17,
  'Truck':18,
  'Van':19,
  'Wheelbarrow':20,
}

MODEL_MAP_SET_B = {
    "Light (Faster)": ("yolo", "dependencies/set_b/yolov10n-best.pt"),
    "Medium (Balanced)": ("detr", "dependencies/set_b/rtdetr-l-best.pt"),
    "Heavy (Better Accuracy)": ("yolo", "dependencies/set_b/yolov10x-best.pt"),
}

SET_B_CLS_MAP = {
    'Bike':0,'Auto':1,'Car':2,'Truck':3,'Bus':4,'Other Vehicle':5
}


MODEL_MAP_SET_C = {
    "Light (Faster)": ("yolo", "dependencies/set_c/yolov10n.pt"),
    "Medium (Balanced)": ("detr", "dependencies/set_c/rtdetr-l.pt"),
    "Heavy (Better Accuracy)": ("yolo", "dependencies/set_c/yolov10x.pt"),
}

SET_C_CLS_MAP = {
    "Person": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorbike": 3,
    "Bus": 5,
    "Truck": 8,
}

MODEL_MAP_SET_D = {
    "Light (Faster)": ("yolo", "dependencies/set_d/yolov8n-obb.pt"),
    "Medium (Balanced)": ("detr", "dependencies/set_d/yolov8m-obb.pt"),
    "Heavy (Better Accuracy)": ("yolo", "dependencies/set_d/yolov8x-obb.pt"),
}

SET_D_CLS_MAP = {
    "Person": 0,
    "Bicycle": 1,
    "Car": 2,
    "Motorbike": 3,
    "Bus": 5,
    "Truck": 8,
}

MODEL_MAP_SET_TEXT = {
    "Small": "dependencies/set_e/yolov8s-world.pt",
    "Medium ": "dependencies/set_e/yolov8m-world.pt",
    "Large": "dependencies/set_e/yolov8l-world.pt",
    "Extra Large": "dependencies/set_e/yolov8x-world.pt",
}   

MODEL_SETS = {
    "CCTV Set A": (MODEL_MAP_SET_A, SET_A_CLS_MAP),
    "CCTV Set B": (MODEL_MAP_SET_B, SET_B_CLS_MAP),
    "Fisheye Camera": (MODEL_MAP_SET_C, SET_C_CLS_MAP),
    "Aerial Camera": (MODEL_MAP_SET_C, SET_C_CLS_MAP),
}

MODEL_SET_ACCIDENT_CLS_MAP = {
    "BIKE":0,
    "BIKE BIKE ACCIDENT":1,
    "BIKE OBJECT ACCIDENT":2,
    "BIKE PERSON ACCIDENT":3,
    "CAR":4,
    "CAR BIKE ACCIDENT":5,
    "CAR CAR ACCIDENT":6,
    "CAR OBJECT ACCIDENT":7,
    "CAR PERSON ACCIDENT":8,
    "PERSON":9,
}

MODEL_SET_ONLY_ACCIDENT_CLS_MAP = {
    "BIKE BIKE ACCIDENT":1,
    "BIKE OBJECT ACCIDENT":2,
    "BIKE PERSON ACCIDENT":3,
    "CAR BIKE ACCIDENT":5,
    "CAR CAR ACCIDENT":6,
    "CAR OBJECT ACCIDENT":7,
    "CAR PERSON ACCIDENT":8,
}

MODEL_SET_ACCIDENT = {
    "Small": "dependencies/set_f/yolov10n-best.pt",
}

SUPERVISION_ANNOTATORS = {
    "Boxes": sv.BoundingBoxAnnotator(),
    "Corners": sv.BoxCornerAnnotator(),
    "Color": sv.ColorAnnotator(),
    "Ellipse": sv.EllipseAnnotator(),
    "Trace": sv.TraceAnnotator(trace_length=100),
    "Circle": sv.CircleAnnotator(),
    "Dot": sv.DotAnnotator(),
    "Heatmap": sv.HeatMapAnnotator(radius=5, opacity=0.2, position=sv.Position.CENTER),
    "Percentage Bar" : sv.PercentageBarAnnotator(),
}

#------------ADDITION FOR ENHANCEMENT------------------
ENHANCEMENT_MODES = ["Off", "On", "Auto"]
DEFAULT_ENHANCEMENT_THRESHOLD = 30
DEFAULT_ENHANCEMENT_MODE = "Off"

# Enhancement models
ENHANCEMENT_MODELS = {
    "Default": {"name": "Default Enhancement", "weight_path": "dependencies/enhancement/zerodce/weight/Epoch99.pth"},
    # Add more models here in the future
    # "Model2": {"name": "Night Vision", "weight_path": "dependencies/enhancement/zerodce/weight/other_model.pth"},
}