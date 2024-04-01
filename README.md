# Food Quality Monitoring System in Food Storage Warehouse

## Index

- [Introduction](#introduction)

- [Demo](#demo)

- [Features](#features)

- [Implementation Details](#implementation-details)

- [Technologies Used](#Technologies-Used)

- [Installation](#installation)

- [Contributors](#Contributors)


## Introduction
This project aims to develop a Food Quality Monitoring System utilizing YOLOv8, a state-of-the-art object detection model, to enhance the efficiency and accuracy of monitoring processes in food storage warehouses. The system provides real-time monitoring of food quality attributes, enabling prompt action in case of any deviations or anomalies.

The process involves collecting data from various online sources, annotating the data with the help of Roboflow, training the model using YOLOv8 architecture, and deploying the trained model on the Streamlit web framework for real-time detection of fresh and rotten food items.

<a name="introduction"></a>

## Demo

 

<a name="demo"></a>

## Features

<a name="features"></a>

#### Real-time Monitoring:
The system provides real-time monitoring of food quality attributes, enabling prompt action in case of any deviations or anomalies.

#### Accuracy and Efficiency: 
YOLOv8's high accuracy and efficiency ensure reliable detection of quality issues while minimizing computational resources.

#### Accessibility:
The web interface allows users to remotely access monitoring data, facilitating seamless oversight of multiple storage locations.

## Key Steps:

<a name="implementation-details"></a>

#### Data Collection:
Data is collected from the internet, comprising images of various food items including both fresh and rotten ones.

#### Data Annotation:    
The collected data is annotated using Roboflow, an annotation platform that streamlines the process of labeling images for object detection tasks.

#### Training:
The annotated data is then utilized to train a YOLOv8 model, a state-of-the-art object detection architecture, using Google Colab. The model is trained to accurately identify fresh and rotten food items in images.

#### Deployment:
After successful training, the trained model is saved as a .pt file. This file is then deployed on the Streamlit web framework,  allowing users to upload images and receive real-time predictions regarding the quality of the food items depicted.

## Technologies Used:

<a name="Technologies-Used"></a>

#### Roboflow: 
Utilized for data annotation, streamlining the process of labeling images for object detection tasks.

#### Google Colab: 
Used for training the YOLOv8 model, taking advantage of its free GPU resources for accelerated model training.

#### YOLOv8: 
Chosen for its efficiency and accuracy in object detection tasks, particularly well-suited for real-time applications.

#### Streamlit: 
Deployed as the web framework for hosting the trained model, providing an intuitive interface for users to interact with the detection system.


  ## Installation
  
  Install the required dependencies:
  
joblib~=1.3.2

opencv-python~=4.9.0.80

numpy~=1.26.4

streamlit~=1.31.1

pillow~=10.2.0

ultralytics~=8.1.10

cvzone~=1.6.1

<a name="installation"></a>

  ## Contributors
  
  [Akshata Jadhav](https://github.com/Akshata196)
  
  [Aditya Mallesh](https://github.com/Adiii89)
  
<a name="Contributors"></a>






