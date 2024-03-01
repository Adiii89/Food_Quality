from ultralytics import YOLO
import cv2
import cvzone
import math
import time
import streamlit as st

def load_yolov8_process_each_Image(image, confidence, st):

    model = YOLO("Food_Quality/Final.pt")
    classNames =  [
    "Apple",
    "Banana",
    "Bellpepper",
    "Bread",
    "Broccoli",
    "Cabbage",
    "Carrot",
    "Cauliflower",
    "Coriander",
    "Egg",
    "Grapes",
    "Kiwi",
    "MBanana",
    "Orange",
    "Papaya",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "RApple",
    "RBanana",
    "RBread",
    "RCauliflower",
    "RCoriander",
    "RGrapes",
    "RGuava",
    "ROrange",
    "RPapaya",
    "RTomato",
    "Strawberry",
    "Tomato"
]


    results = model(image, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box Code
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(image, (x1, y1, w, h))

            # confidence Value Code
            conf = math.ceil((box.conf[0]*100))/100

            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1)
    st.subheader('Output Image')
    st.image(image, channels='BGR', use_column_width=True)


def load_yolov8_process_each_Frame(video_name, kpi_text, kpi2_text, kpi3_text, stframe):
    cap = cv2.VideoCapture(video_name)
    width=cap.set(3, 1280)
    height=cap.set(4, 720)

    model = YOLO("Food_Quality/Final.pt")
    prev_time=0
    classNames = [
    "Apple",
    "Banana",
    "Bellpepper",
    "Bread",
    "Broccoli",
    "Cabbage",
    "Carrot",
    "Cauliflower",
    "Coriander",
    "Egg",
    "Grapes",
    "Kiwi",
    "MBanana",
    "Orange",
    "Papaya",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "RApple",
    "RBanana",
    "RBread",
    "RCauliflower",
    "RCoriander",
    "RGrapes",
    "RGuava",
    "ROrange",
    "RPapaya",
    "RTomato",
    "Strawberry",
    "Tomato"
]
    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Bounding Box Code
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))

                # confidence Value Code
                conf = math.ceil((box.conf[0] * 100)) / 100
                print(conf)

                # Class Name
                cls = int(box.cls[0])

                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1)
            stframe.image(img, channels='BGR', use_column_width=True)
            current_time=time.time()
            fps=1/(current_time-prev_time)

            prev_time=current_time
            kpi_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(fps)}</h1>", unsafe_allow_html=True)
            kpi2_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(width)}</h1>", unsafe_allow_html=True)
            kpi3_text.write(f"<h1 style='text-align:center; color:red;'>{'{:.1f}'.format(height)}</h1>", unsafe_allow_html=True)
