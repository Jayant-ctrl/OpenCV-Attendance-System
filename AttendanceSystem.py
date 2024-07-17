import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'images/img_attendance'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

def findEncodings(images):
   encodeList = []
   for img in images:
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)       
      encode = face_recognition.face_encodings(img)[0] 
      encodeList.append(encode)      
   return encodeList

def markAttendance(name):
  with open('Attendance.csv', 'r+') as f:
      myDataList = f.readlines()
      nameList = []
      for line in myDataList:
         entry = line.split(',')
         nameList.append(entry[0])

         if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dtString}')           


encodeListKnown = findEncodings(images)
print('Encodings Complete')      

capture =cv2.VideoCapture(0)
while True:
   isTrue, img = capture.read()
   imgSmall = cv2.resize(img, (0,0), None, 0.25, 0.25)
   imgSmall = cv2.cvtColor(imgSmall, cv2.COLOR_BGR2RGB)

   facesCurFrame = face_recognition.face_locations(imgSmall)
   encodingCurFrame = face_recognition.face_encodings(imgSmall, facesCurFrame)

   for encodeFace,faceLoc in zip(encodingCurFrame, facesCurFrame):
       matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
       faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
       print(faceDis)
       matchIndex = np.argmin(faceDis)

       if matches[matchIndex]:
          name = classNames[matchIndex]
          print(name)
          y1,x2,y2,x1 = faceLoc
          y1,x2,y2,x1 = y1*4, x2*4, y2*4, x1*4
          cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
          cv2.rectangle(img, (x1,y2-35), (x2,y2), (0,255,0), -1)
          cv2.putText(img, name, (x1+6,y2-6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1)
          markAttendance(name)

   cv2.imshow('Webcam', img)
   cv2.waitKey(1)










