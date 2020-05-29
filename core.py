#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Required Downloads
"""
    libraries down
    facenet_keras model : https://drive.google.com/drive/folders/12aMYASGCKvDdkygSv1yQq8ns03AStDO_
    haarcascade opencv classifier : https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

"""


import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
import os
import time


# In[2]:


# define facedetector and model , both takes time
model=keras.models.load_model('facenet_keras.h5',compile=False)
facedetector=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


# In[3]:


def add_student_face_recognition(directory,name):
    try:
        os.mkdir(os.path.join(directory,name))
    except OSError as error:
        print('Student already exists')
        return 
    
    try:
        cam=cv2.VideoCapture(0)
        pic_count=-2
        while pic_count<50:
            ret,frame=cam.read()
            if ret==0:
                break 
            faces=facedetector.detectMultiScale(
                image=frame,
                scaleFactor=1.1,
                minNeighbors=8
            )
            for (x,y,w,h) in faces:
                pic_count+=1
                if pic_count>0:
                    cv2.imwrite(os.path.join(os.path.join(directory,name),str(pic_count)+'.jpg'),frame[y:y+h,x:x+w])
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
                
            cv2.imshow('img',frame)
            if cv2.waitKey(500)&0xff==ord('q'):
                break
    except:
        print('Erro while capturing images')
    finally:
        cam.release()
        cv2.destroyAllWindows()


# In[4]:


def get_students_encoding(model,directory):
    students_encoding={}
    list_of_students=os.listdir(directory)
    for student in list_of_students:
        images=[]
        newdirectory=os.path.join(directory,student)
        for pic in os.listdir(newdirectory):
            image=cv2.imread(os.path.join(newdirectory,pic))
            images.append(cv2.resize(image,(160,160),cv2.INTER_AREA)/255.0)
        dataset=np.array(images)
        encode=model.predict(dataset)
        students_encoding[student]=encode
    
    return students_encoding


# In[5]:


def calculatedistance(encode1,encode2):
    distance=tf.reduce_sum(tf.square(tf.subtract(encode1,encode2)))
    return distance


# In[6]:


def get_min_distance(test_encoding,list_of_encodings):
    dist=1000
    for encoding in list_of_encodings:
        dist=min(dist,np.linalg.norm(test_encoding-encoding))
        #dist=min(dist,calculatedistance(test_encoding,encoding))
    return dist


# In[7]:


def identify_student(test_encoding,students_encoding,threshold):
    dist=1000
    id="unknown"
    
    for name,encodings in students_encoding.items():
        newdist=get_min_distance(test_encoding,encodings)
        if newdist<dist:
            dist=newdist
            id=name
    if dist>threshold:
        id="unknown"
    return id


# In[8]:


directory=r'facedatabase'
threshold=8.0


# In[9]:


import datetime
import csv
def take_attendance_face_recognition():
    # Face recognition (with liveness detection) code
    # Generate and store the face encodings of known students using their face images.
    # To be done using pre trained model( FaceNet) but here i used face_recognition python library .
    # Images of students are stored in 'students_images' directory .
    print(1)
    known_faces_encodings=get_students_encoding(model,directory)

    try:
        video=cv2.VideoCapture(0)
        success=1
        print(1)

        # OpenCV Haar-feature based cascade classifier for detecting eyes 
        eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')

        # load the model trained for predicting whether or not the eyes are closed
        json_file=open("eye_model.json","r")
        eye_model=tf.keras.models.model_from_json(json_file.read())
        json_file.close()
        eye_model.load_weights("eye_model.h5")


        # Dictionary with key representing the rollno of recognised student and value representing the status of eyes ( either open or closed )
        recognised_students={}

        while success:

            sucess,frame=video.read()
            #print(frame.shape)
            frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            is_processed=0
            if is_processed==0 :


                #detected_faces_locations = fc.face_locations(frame)
                detected_faces_locations=facedetector.detectMultiScale(frame,1.1,8)
               # detected_faces_encodings = fc.face_encodings(frame,detected_faces_locations)

                detected_faces_rollno=[]

                # For each detected face in the frame generate its face encoding and compare it with encodings of known students.
                # If matched append its rollno in an array or append 'unknown'
                #for encoding in detected_faces_encodings:

                #    rollno=identify_student(encoding,known_faces_encodings,threshold)

                #    detected_faces_rollno.append(rollno)

                for (x,y,w,h) in detected_faces_locations:
                    croppedframe=frame[y:y+h,x:x+w]
                    croppedframe=cv2.resize(croppedframe,(160,160),cv2.INTER_AREA)/255.0
                    dataset=np.array([croppedframe])
                    encoding_of_frame=model.predict(dataset)[0]
                    rollno=identify_student(encoding_of_frame,known_faces_encodings,threshold)

                    detected_faces_rollno.append(rollno)



                #frame = cv2.putText(frame,id,(x,y),cv2.FONT_HERSHEY_SIMPLEX ,1,(255,255,255),1, cv2.LINE_AA)

                # For each detected Face do 
                for (fx,fy,fw,fh),rollno in zip(detected_faces_locations,detected_faces_rollno):
                    left=fx 
                    right=fx+fw
                    top=fy
                    bottom=fy+fh

                    if rollno=='unknown':
                        # if person is unknown do noting
                        cv2.rectangle(frame,(left,top),(right,bottom),(0,0,255),2)
                    else:
                        # if student is known do liveness detection using blink detection
                        cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),2)
                        cv2.putText(frame,rollno,(left+6,bottom-6),cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),1)

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

                            classes=eye_model.predict(image_to_classify,batch_size=10)

                            # If probab that eye is open is greater than 0.5 predict eye is open
                            if classes[0][0]>=0.5 :
                                flag=1
                                
                            cv2.rectangle(frame,(ex+left,ey+top),(ex+ew+left,ey+eh+top),(0,255,0),2)

                        status='Closed'
                        if flag==1:
                            status='Open'

                       
                        cv2.putText(frame,status,(left+6,bottom+20),cv2.FONT_HERSHEY_DUPLEX,1.0,(0,0,255),1)

                        # If eyes are closed now but previously it was observed to be open->Blink detected
                        if status=='Closed':
                            if rollno not in recognised_students:
                                recognised_students.update({rollno:0})
                            else:
                                if recognised_students[rollno]==1:
                                    #---------------------------------------
                                    #print('Present '+rollno+' '+str(time.ctime()))
                                    now=datetime.datetime.now()
                                    date_now=now.strftime("%Y-%m-%d")
                                    time_now=now.strftime("%H:%M:%S")
                                    attendance_today = 'Attendance/'+date_now+'.csv'
                                    is_present=0 
                                    try:
                                        
                                        if os.path.isfile(attendance_today)==False:
                                            with open(attendance_today,"w",newline='') as csvfile :
                                                writer=csv.writer(csvfile)
                                                

                                        with open(attendance_today,"r") as csvfile:
                                            reader=csv.reader(csvfile)
                                            for row in reader:
                                                if row[0]==rollno:
                                                    is_present=1


                                        if is_present==0:
                                            with open(attendance_today,"a+") as csvfile:
                                                writer=csv.writer(csvfile)
                                                writer.writerow([rollno])
                                    
                                    except:
                                        print('not able to write to file')
                                    
                                        
                                    
                                    
                                    #---------------------------------------
                                recognised_students[rollno]=0

                        else:
                            if rollno not in recognised_students:
                                recognised_students.update({rollno:1})
                            else:
                                recognised_students[rollno]=1



                cv2.imshow('Videos',frame)

                if cv2.waitKey(1)&0xff==ord('q'):
                    break
                success=1
                is_processed=1

        cv2.destroyAllWindows()
        video.release()
    
    except :
        print('error')
    finally:
        video.release()
        cv2.destroyAllWindows()



if __name__=="__main__":
    take_attendance_face_recognition()


