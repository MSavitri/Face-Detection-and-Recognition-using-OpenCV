# Face-Detection-and-Recognition-using-OpenCV
from sklearn.neighbors import KNeighborsClassifier import cv2 
# Step 1 : importing the K-Nearest Neighbors (KNN) classifier from scikit-learn and importing cv2, which is part of OpenCV for computer vision tasks. 
K-Nearest Neighbors (KNN) is a simple, yet powerful, supervised machine learning algorithm commonly used for classification and regression tasks. It makes predictions based on the similarity of input data points to other data points in the training set.
import pickle 
Step 2 : The pickle module in Python is used to serialize (save) and deserialize (load) Python objects. Serialization is the process of converting a Python object into a byte stream, while deserialization is the reverse process. Here we use Deserialization
import numpy as np 
Step 3 : The numpy library in Python is a powerful tool for numerical computing. It provides support for arrays, matrices, and high-level mathematical functions to operate on these data structures efficiently.
import os
Step 4 : The os module in Python provides a way to interact with the operating system. It offers functionality for file and directory manipulation, environment management, and other system-level operations.
video=cv2.VideoCapture(0) 
Opens a webcam feed using cv2.VideoCapture(0)
facedetect=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
Uses a pre-trained Haar Cascade classifier (haarcascade_frontalface_default.xml) to detect faces in the video frames.
with open('data/names.pkl', 'rb') as f: 
    LABELS=pickle.load(f) 
with open('data/faces_data.pkl', 'rb') as f:
    FACES=pickle.load(f)
Loads serialized data using pickle: LABELS: Labels for the faces (e.g., person names).FACES: Face embeddings (feature data) for training.
knn=KNeighborsClassifier(n_neighbors=5) 
knn.fit(FACES, LABELS)
Trains a K-Nearest Neighbors (KNN) model on the pre-loaded face embeddings and their corresponding labels.
while True: 
    ret,frame=video.read()
Capture a single frame from the video feed
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
Convert the frame to grayscale
    faces=facedetect.detectMultiScale(gray, 1.3 ,5) 
Detect faces in the frame
    for (x,y,w,h) in faces: 
Loop through all detected faces
        crop_img=frame[y:y+h, x:x+w, :]
Crop the region containing the face
        resized_img=cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)  
Resize and flatten the image
        output=knn.predict(resized_img)
Predict the label of the face using KNN
        cv2.putText(frame, str(output[0]), (x,y-15), 
        cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 1) 
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)  
Annotate the frame with the predicted label and Draw a rectangle around the face
   cv2.imshow("Frame",frame)   
Display the annotated frame in a window
   k=cv2.waitKey(1) 
   if k==ord('q'): 
      break
cv2.waitKey(1) listens for key presses. If the 'q' key is pressed, the loop breaks, and the program exits.
video.release() 
cv2.destroyAllWindows()
closes all the Windows
