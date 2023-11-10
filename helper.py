from ultralytics import YOLO
import time
import streamlit as st
import cv2
import csv
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pytube import YouTube

import settings


def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model

def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)
        #print(res)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()    
    boxes = res[0].boxes

    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

    with open('result.csv', 'a', newline='') as file:
        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            conf = int(box.conf[0]*100)
            row = {'class_id':str(cls), 'class_name': class_name, 'total': 1, 'confidence': str(conf)}
            writer = csv.DictWriter(file, fieldnames=row.keys())
            writer.writerow(row)

def display_reports():
    try:
        with open('result.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)

        total_alligator_cracking = 0
        total_longitudinal_cracking = 0
        total_lateral_cracking = 0
        total_pothole = 0

        for item in data:
            if item[1] == "alligator cracking":
                total_alligator_cracking +=1
            elif item[1] == "longitudinal cracking":
                total_longitudinal_cracking +=1
            elif item[1] == "lateral cracking":
                total_lateral_cracking +=1
            elif item[1] == "pothole":
                total_pothole +=1
        
        st.metric(label="Alligator Cracking", value=total_alligator_cracking)
        st.metric(label="Longitudinal Cracking", value=total_longitudinal_cracking)
        st.metric(label="Lateral Cracking", value=total_lateral_cracking)
        st.metric(label="Pothole", value=total_pothole)
    except Exception as e:
        st.write(e)

def clean_reports():
    try:
        filename = "result.csv"
        # opening the file with w+ mode truncates the file
        f = open(filename, "w+")
        f.close()
    except Exception as e:
        print(e)

def play_youtube_video(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_youtube = st.sidebar.text_input("YouTube Video url")
    #is_display_tracker, tracker = display_tracker_options()

    if st.sidebar.button('Detect Objects'):        
        try:
            clean_reports()
            yt = YouTube(source_youtube)
            stream = yt.streams.filter(file_extension="mp4", res=720).first()
            vid_cap = cv2.VideoCapture(stream.url)

            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             True,
                                             "bytetrack.yaml",
                                             )                       
                else:
                    vid_cap.release()
                    break
                        
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

    if st.sidebar.button('Display Reports'):
        display_reports()

def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    #is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            clean_reports()
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             True,
                                            "bytetrack.yaml"
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))
    if st.sidebar.button('Display Reports'):
        display_reports()

def play_webcam(conf, model):
    """
    Plays a webcam stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_webcam = settings.WEBCAM_PATH
    #is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            clean_reports()
            vid_cap = cv2.VideoCapture(source_webcam)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             True,
                                             "bytetrack.yaml",
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    if st.sidebar.button('Display Reports'):
        display_reports()

def play_stored_video(conf, model, source_vid):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    #source_vid = st.sidebar.selectbox(
    #    "Choose a video...", settings.VIDEOS_DICT.keys())

    #is_display_tracker, tracker = display_tracker_options()

    #with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
    #    video_bytes = video_file.read()
    #if video_bytes:
    #    st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            clean_reports()
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             True,
                                             "bytetrack.yaml"
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))
    if st.sidebar.button('Display Reports'):
        display_reports()