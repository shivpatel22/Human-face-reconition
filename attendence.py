import csv
import os
from datetime import datetime
from datetime import date

import cv2
import face_recognition
import numpy as np

path = 'images'
flag = 0
images = []
personName = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
print(personName)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = faceEncodings(images)
print("All Encodings Complete!!")


def attendence(name):
    with open('attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])

        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{name},{tStr},{dStr}' + "\n")


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
    if flag == 0:
        flag = flag + 1
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            # print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            attendence(name)

        else:

            if flag > 0:
                with open('attendence.csv', 'w+') as f:
                    myDataList = f.writelines()
                    nameList = []
                    for line in myDataList:
                        print("Match not found !")
                        exit(0)

    i = cv2.resize(frame, (1200, 660))
    cv2.imshow("Camera", i)
    if cv2.waitKey(10) == 13:
        current_date = str(date.today())
        file_name = str(current_date + '.csv')
        with open("attendence.csv", "r") as fp1:
            with open(file_name, "w") as fp2:
                rs = csv.reader(fp1)
                wp = csv.writer(fp2, delimiter=',')
                for i in rs:
                    wp.writerow(i)
        exit(0)
        break
cap.release()
cv2.destroyAllWindows()