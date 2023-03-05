{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "cf00179a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['jobs.jpg', 'kalam.jpg', 'Keyur.jpg', 'Milan.jpg', 'Shiv.jpg', 'Shivani.jpg', 'tata.jpg', 'tesla.jpeg']\n",
      "['jobs', 'kalam', 'Keyur', 'Milan', 'Shiv', 'Shivani', 'tata', 'tesla']\n",
      "All Encodings Complete!!\n"
     ]
    }
   ],
   "source": [
    "import cv2\n",
    "import numpy as np \n",
    "import face_recognition\n",
    "import os\n",
    "from datetime import datetime\n",
    "\n",
    "path = 'images'\n",
    "images = []\n",
    "personName = []\n",
    "myList = os.listdir(path)\n",
    "print(myList)\n",
    "for cu_img in myList: \n",
    "    current_Img = cv2.imread(f'{path}/{cu_img}')\n",
    "    images.append(current_Img)\n",
    "    personName.append(os.path.splitext(cu_img)[0])\n",
    "print(personName)\n",
    "\n",
    "def faceEncodings(images):\n",
    "    encodeList = []\n",
    "    for img in images:\n",
    "        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
    "        encode = face_recognition.face_encodings(img)[0]\n",
    "        encodeList.append(encode)\n",
    "    return encodeList\n",
    "\n",
    "encodeListKnown = faceEncodings(images)\n",
    "print(\"All Encodings Complete!!\")\n",
    "\n",
    "def attendence(name):\n",
    "    with open('attendence.csv', 'r+') as f:\n",
    "        myDataList = f.readlines()\n",
    "        nameList = []\n",
    "        for line in myDataList:\n",
    "            entry = line.split(',')\n",
    "            nameList.append(entry[0])\n",
    "\n",
    "        if name not in nameList:\n",
    "            time_now = datetime.now()\n",
    "            tStr = time_now.strftime('%H:%M:%S')\n",
    "            dStr = time_now.strftime('%d/%m/%Y')\n",
    "            f.writelines(f'{name},{tStr},{dStr}')\n",
    "\n",
    "\n",
    "cap = cv2.VideoCapture(0)\n",
    "\n",
    "while True:\n",
    "    ret, frame = cap.read()\n",
    "    faces = cv2.resize(frame, (0,0), None, 0.25, 0.25)\n",
    "    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)\n",
    "\n",
    "    facesCurrentFrame = face_recognition.face_locations(faces)\n",
    "    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)\n",
    "\n",
    "    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):\n",
    "        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)\n",
    "        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)\n",
    "\n",
    "        matchIndex = np.argmin(faceDis)\n",
    "\n",
    "        if matches[matchIndex]:\n",
    "            name = personName[matchIndex].upper()\n",
    "           # print(name)\n",
    "            y1,x2,y2,x1 = faceLoc\n",
    "            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4\n",
    "            cv2.rectangle(frame, (x1,y1),(x2,y2), (0,255,0), 2)\n",
    "            cv2.rectangle(frame, (x1, y2-35),(x2,y2),(0,255,0), cv2.FILLED)\n",
    "            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)\n",
    "            attendence(name)\n",
    "\n",
    "    cv2.imshow(\"Camera\", frame)\n",
    "    if cv2.waitKey(10) == 13:\n",
    "        break\n",
    "cap.release()\n",
    "cv2.destroyAllWindows()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3fe9c007",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
