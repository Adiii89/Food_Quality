from ultralytics import YOLO
import cv2
import cvzone
import math

image=cv2.imread('../Images12/15.jpg')


model = YOLO("/FoodQualitiy.pt")
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
    "Pineapple",
    "Pomegranate",
    "Potato",
    "RApple",
    "RBanana",
    "RBread",
    "RCauliflower",
    "RCoriander",
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
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(image, (x1, y1, w, h))

            # confidence Value Code
            conf = math.ceil((box.conf[0]*100))/100
            print(conf)

            # Class Name
            cls = int(box.cls[0])

            cvzone.putTextRect(image, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.9, thickness=1)


cv2.imshow("Image", image)
cv2.waitKey(1)
