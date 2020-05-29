# Face recognition (with liveness detection) code

import face_recognition as fc 
import cv2 
import os 
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import time



# Generate and store the face encodings of known students using their face images.
# To be done using pre trained model( FaceNet) but here i used face_recognition python library .
# Images of students are stored in 'students_images' directory .


known_faces_images = os.listdir('students_images')

known_faces_encodings = []
known_faces_rollnos = []


def get_rollno(image):
    rollno=''
    for i in range(len(image)):
        if image[i]=='.':
            break
        rollno=rollno+image[i]
    return rollno
        
for image in known_faces_images :
    current_image = fc.load_image_file('students_images/'+image)
    current_image_encodings = fc.face_encodings(current_image)[0]
    current_image_rollno = get_rollno(image)

    known_faces_encodings.append(current_image_encodings)
    known_faces_rollnos.append(current_image_rollno)


video=cv2.VideoCapture(0)

success=1

# OpenCV Haar-feature based cascade classifier for detecting eyes 
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

# load the model trained for predicting whether or not the eyes are closed
json_file=open("eye_model.json","r")
model=tf.keras.models.model_from_json(json_file.read())
json_file.close()
model.load_weights("eye_model.h5")


# Dictionary with key representing the rollno of recognised student and value representing the status of eyes ( either open or closed )
recognised_students={}

while success:

    sucess,frame=video.read()
    if success ==1 :
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    else:
        continue
    is_processed=0
    if is_processed==0 :
        

        detected_faces_locations = fc.face_locations(frame)
        detected_faces_encodings = fc.face_encodings(frame,detected_faces_locations)

        detected_faces_rollno=[]

        # For each detected face in the frame generate its face encoding and compare it with encodings of known students.
        # If matched append its rollno in an array or append 'unknown'
        for encoding in detected_faces_encodings:
            matches = fc.compare_faces(known_faces_encodings,encoding)
            rollno='unknown'
            distances = fc.face_distance(known_faces_encodings,encoding)
            
            potential_candidate = np.argmin(distances)

            if matches[potential_candidate] :
                rollno=known_faces_rollnos[potential_candidate]
                
            detected_faces_rollno.append(rollno)
            
        # For each detected Face do 
        for (top,right,bottom,left),rollno in zip(detected_faces_locations,detected_faces_rollno):
            
            if rollno=='unknown':
                # if person is unknown do noting
                cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
            else:
                # if student is known do liveness detection using blink detection
                cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)

                # detect eyes in the face
                roi_gray = frame_gray[top:bottom,left:right]
                eyes= eye_cascade.detectMultiScale(roi_gray)

                # If any one or both the eyes are open flag becomes 1
                flag=0
                for (ex,ey,ew,eh) in eyes:
                    roi=frame[ey+top:ey+eh+top,ex+left:ex+left+ew]
                    roi=cv2.resize(roi,(32,32),interpolation=cv2.INTER_AREA)
                    roi=np.expand_dims(roi,axis=0)
                    roi=np.reshape(roi,(1,32,32,3))
                    image_to_classify=np.vstack([roi])
                    
                    classes=model.predict(image_to_classify,batch_size=10)

                    # If probab that eye is open is greater than 0.5 predict eye is open
                    if classes[0][0]>=0.5 :
                        flag=1
                    cv2.rectangle(frame,(ex+left,ey+top),(ex+ew+left,ey+eh+top),(0,255,0),2)

                status='Closed'  
                if flag==1:
                    status='Open'

                    
                cv2.putText(frame,rollno,(left+6,bottom-6),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)
                cv2.putText(frame,status,(left+6,bottom+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,0,0),1)

                # If eyes are closed now but previously it was observed to be open->Blink detected
                if status=='Closed':
                    if rollno not in recognised_students:
                        recognised_students.update({rollno:0})
                    else:
                        if recognised_students[rollno]==1:
                            #---------------------------------------
                            print('Present '+rollno+' '+str(time.ctime()))
                            #---------------------------------------
                            recognised_students[rollno]=0

                else:
                    if rollno not in recognised_students:
                        recognised_students.update({rollno:1})
                    else:
                        recognised_students[rollno]=1
                

      
        cv2.imshow('Videos',frame)
        cv2.waitKey(1)
        
        success=1
        is_processed=1

cv2.destroyAllWindows()
video.release()






    
        

