from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

cap=cv2.VideoCapture('cars.mp4')
cap.set(3,640)
cap.set(4,480)

model=YOLO('yolov8n.pt')

mask=cv2.imread('mask_1.png')

tracker=Sort(max_age=20,min_hits=3,iou_threshold=0.3)

totalcnt=[]

limits=[400,297,673,297]

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

while(True):
    success,img=cap.read()
    imgregion=cv2.bitwise_and(mask,img)
    imggraphics=cv2.imread("graphics_1.png",cv2.IMREAD_UNCHANGED)
    img=cvzone.overlayPNG(img,imggraphics,(0,0))
    results=model(imgregion,stream=True)
    detections=np.empty((0, 5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            #Bounding boxes
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            w,h=x2-x1,y2-y1
            
            #confidence
            conf=(math.ceil(box.conf[0]*100))/100
            #conf=str(conf)
            #cvzone.cornerRect(img,(x1,y1,w,h))
            #cv2.putText(img,conf,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2,cv2.LINE_AA)
            #Classes
            cls=int(box.cls[0])
            #cv2.putText(img,str(classNames[cls]),(x1,y1+10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1,cv2.LINE_AA)

            if classNames[cls]=="car" or classNames[cls]=="truck" or classNames[cls]=="motorbike" and conf>0.3:
                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                #cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1,offset=3)
                #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                curarr=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,curarr))
                



    restracker=tracker.update(detections)

    cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,0,255),3)

    for result in restracker:
      x1,y1,x2,y2,Id=result
      x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
      print(x1,y1,x2,y2,Id)
      w,h=x2-x1,y2-y1
      cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2,colorR=(255,0,0))
      cvzone.putTextRect(img, f'{Id}', (max(0, x1), max(35, y1)), scale=2, thickness=3,offset=10)
      cx,cy=x1+w/2,y1+h/2
      if limits[0]<cx<limits[2] and limits[1]-20<cy<limits[1]+20:
          if(totalcnt.count(Id)==0):
            totalcnt.append(Id)
            cv2.line(img,(limits[0],limits[1]),(limits[2],limits[3]),(0,255,0),5)

      #cvzone.putTextRect(img, f'Count:{len(totalcnt)}', (50,50))
      cv2.putText(img,str(len(totalcnt)),(255,100),cv2.FONT_HERSHEY_PLAIN,5,(50,50,255),8)
          

    cv2.imshow('Image',img)
    cv2.imshow('ImageRegion',imgregion)
    if(cv2.waitKey(10)&0xFF==ord('d')):
        break




cap.release()
cv2.destroyAllWindows()
