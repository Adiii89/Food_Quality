import cv2
import numpy as np
import streamlit as st
from VidEo_Streamlit import *
from PIL import Image
import base64
import tempfile
def main():
     st.title("Food Quality Monitoring System in Food Storage warehouse ")
     st.sidebar.title("Settings")
     st.sidebar.subheader("Parameters")


     st.markdown(
         """
         <style>
         [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
         width: 300px;
         }
         
         [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
         width: 300px;
         margin-left:-300px;
         }
         
            [data-testid="stImage"][aria-expanded="true"]>div:first-child{
         width: 1600px;
         height: 900px;
         
         }
         
         
         </style>
         """,
         unsafe_allow_html=True,

     )

     app_mode = st.sidebar.selectbox('Choose the App mode', ['About App', 'Run on Image', 'Run on Video'])

     if app_mode=='About App':
         st.header('Introduction ', divider='rainbow')
         with st.container():

             st.write(
                 "Food quality and safety are paramount concerns in the food industry, particularly in storage warehouses where large quantities of perishable goods are stored. Traditional monitoring methods often rely on manual inspection, which can be time-consuming and prone to errors. To address these challenges, the implementation of automated systems utilizing advanced technologies such as deep learning has gained prominence.")

         st.image('C:/Users/admin/PycharmProjects/pythonProject1/Food_Quality/Images12/ids-nxt-industrial-camera-artificial-intelligence-apple-detection-e4d34ebc.jpg')

         st.markdown(
             """
             <style>
             [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
             width: 300px;
             }

             [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
             width: 300px;
             margin-left:-300px;
             }
            
            [data-testid="stImage"][aria-expanded="true"]>div:first-child{
             width: 1600px;
             height:900px;
             
             }
             </style>
             """,
             unsafe_allow_html=True,

         )


         st.markdown('''
                     The primary objective of this project is to develop a Food Quality Monitoring System utilizing YOLOv8, a state-of-the-art object detection model, to enhance the efficiency and accuracy of monitoring processes in food storage warehouses.
                     ''')
         st.header('Key Features', divider='rainbow')
         st.write(" **1.Real-time Monitoring:** "
                  "The system provides real-time monitoring of food quality attributes, enabling prompt action in case of any deviations or anomalies.")
         st.write(" **2.Accuracy and Efficiency:** "
                  "YOLOv8's high accuracy and efficiency ensure reliable detection of quality issues while minimizing computational resources. ")
         st.write(" **3.Accessibility:** "
                  "The web interface allows users to remotely access monitoring data, facilitating seamless oversight of multiple storage locations.")
         st.image('C:/Users/admin/PycharmProjects/pythonProject1/Food_Quality/Images12/banner-integrations.png')
         st.header('Conclusion', divider='rainbow')
         st.write("The integration of YOLOv8 in a Food Quality Monitoring System offers significant advancements in ensuring the safety and quality of stored food products. By automating the monitoring process and providing real-time insights, this system enhances efficiency, reduces costs, and mitigates risks associated with food storage operations."
                  "Future enhancements may involve incorporating additional sensors for multi-modal data fusion, expanding the model's capabilities to recognize a broader range of quality attributes, and integrating predictive analytics for proactive quality management.")

     elif app_mode=='Run on Image':
         st.sidebar.markdown('---')
         confidence=st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0)
         st.sidebar.markdown('---')

         st.markdown(
             """
             <style>
             [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
             width: 300px;
             }

             [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
             width: 300px;
             margin-left:-300px;
             }
             </style>
             """,
             unsafe_allow_html=True,

         )
         img_file_buffer = st.sidebar.file_uploader('Upload an Image', type=["jpg", "jpeg", "png"])
         DEMO_IMAGE = "C:/Users/admin/PycharmProjects/pythonProject1/Food_Quality/Images12/33.jpg"

         if img_file_buffer is not None:
             img = cv2.imdecode(np.fromstring(img_file_buffer.read(), np.uint8), 1)
             image = np.array(Image.open(img_file_buffer))
         else:
             img = cv2.imread(DEMO_IMAGE)
             image = np.array(Image.open(DEMO_IMAGE))
         st.sidebar.text('Original Image')
         st.sidebar.image(image)
         load_yolov8_process_each_Image(img, confidence, st)

     elif app_mode == 'Run on Video':
         st.markdown(
             """
             <style>
             [data-testid="stSidebar"][aria-expanded="true"]>div:first-child{
             width: 300px;
             }

             [data-testid="stSidebar"][aria-expanded="false"]>div:first-child{
             width: 300px;
             margin-left:-300px;
             }
             </style>
             """,
             unsafe_allow_html=True,

         )
         st.sidebar.markdown('---')
         use_webcam=st.sidebar.checkbox('Use Webcam')
         st.sidebar.markdown('---')
         video_file_buffer=st.sidebar.file_uploader('Upload a Video', type=["mp4","avi", "asf", "mov", "webm"])
         DEMO_VIDEO='C:/Users/admin/PycharmProjects/pythonProject1/Food_Quality/Videos/video1.webm'

         tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
         if not video_file_buffer:
             if use_webcam:
                 tffile.name=0
             else:
                 vid =cv2.VideoCapture(DEMO_VIDEO)
                 tffile.name=DEMO_VIDEO
                 demo_vid=open(tffile.name, 'rb')
                 demo_bytes=demo_vid.read()
                 st.sidebar.text('Input Video')
                 st.sidebar.video(demo_bytes)
         else:
             tffile.write(video_file_buffer.read())
             demo_vid = open(tffile.name, 'rb')
             demo_bytes = demo_vid.read()
             st.sidebar.text('Input Video')
             st.sidebar.video(demo_bytes)
         stframe=st.empty()
         st.markdown(" ", unsafe_allow_html=True)
         kpi1, kpi2, kpi3=st.columns(3)
         with kpi1:
             st.markdown("**Frame Rate**")
             kpi_text=st.markdown("0")
         with kpi2:
             st.markdown("**Width**")
             kpi2_text=st.markdown("0")
         with kpi3:
             st.markdown("**Height**")
             kpi3_text=st.markdown("0")
         st.markdown(" ", unsafe_allow_html=True)
         load_yolov8_process_each_Frame(tffile.name, kpi_text, kpi2_text, kpi3_text, stframe)



if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass