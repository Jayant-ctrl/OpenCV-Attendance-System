import cv2
import numpy
import face_recognition


imgRGB = face_recognition.load_image_file('3.jpeg')
imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2RGB)

imgTest = face_recognition.load_image_file('test2.jpeg')
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgRGB)[0]
encodeFace = face_recognition.face_encodings(imgRGB)[0]
print(faceLoc)
cv2.rectangle(imgRGB, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255,0,0), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
print(faceLocTest)
cv2.rectangle(imgTest, (faceLocTest[3], faceLocTest[0]), (faceLocTest[1], faceLocTest[2]), (255,0,0), 2)

results = face_recognition.compare_faces([encodeFace], encodeTest)
faceDis = face_recognition.face_distance([encodeFace], encodeTest)
print(results, faceDis)

cv2.putText(imgTest, f'{results} {round(faceDis[0], 2)}', (10,30), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)

cv2.imshow('load', imgRGB)
cv2.imshow('test', imgTest)
cv2.waitKey(0)