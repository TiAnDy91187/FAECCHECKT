import cv2
import time
import numpy as np
import face_recognition
import os
 
path = "BaseData"
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
encodeListKnown = findEncodings(images)
print('Encoding Complete')

CapCam0 = cv2.VideoCapture(0)

CheckFps = CapCam0.get(cv2.CAP_PROP_FPS)

PicWidth = CapCam0.get(cv2.CAP_PROP_FRAME_WIDTH)
PicHeight = CapCam0.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(CheckFps,"CheckFps")
print(PicWidth,"*",PicHeight)

CapFlag=0
Cnttime=0


while True:
    (ReturnVal,Picture)=CapCam0.read()
    if ReturnVal:

        k=cv2.waitKey(1)
        if k==27:
            break
        elif k==67 or k==99:
            PersonNameInput = input("Let input Name: ")
            PersonNameOutput=str(PersonNameInput) + ".png"
            cv2.imwrite("BaseData/"+PersonNameOutput,Picture,[cv2.IMWRITE_PNG_COMPRESSION,3])
    else:
        print("Camera is not ready")
        break

    imgS = cv2.resize(Picture,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
 
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
 
    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
 #       nametest ="BossOfChi"
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(Picture,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(Picture,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(Picture,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    GetSecTime=int(time.mktime(time.localtime(time.time())))
    if CapFlag==0:
        Cnttime=GetSecTime + 30
        CapFlag=1
    elif Cnttime<=GetSecTime:
        PictureOutput="P"+str(GetSecTime)+ ".png"
        cv2.imwrite("PictureOut/" + PictureOutput,Picture,[cv2.IMWRITE_PNG_COMPRESSION,3])
        CapFlag=0
    cv2.imshow("CamOn: 0", Picture)

CapCam0.release()
cv2.destroyAllWindows()