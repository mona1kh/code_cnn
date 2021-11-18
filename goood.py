import dlib
import cv2
import sys
import numpy as np
from imutils import face_utils
import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import threading
from pygame import mixer
from tensorflow.keras.models import load_model



predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

capture=cv2.VideoCapture(0)
if not (capture.isOpened()):
    print("Error")

model = load_model('__model2_.h5')
mixer.init()
sound = mixer.Sound('alarm.ogg')

class FaceDetector(object):
  #MTCNN Start  
    def __init__(self, mtcnn):
        self.mtcnn = mtcnn
        
    def _draw(self, frame, boxes, probs, landmarks):   
        try:
            for box, prob, ld in zip(boxes, probs, landmarks):    
                cv2.rectangle(frame,
                              (box[0], box[1]),
                              (box[2], box[3]),
                              (0, 0, 255),
                              thickness=2)
                cv2.putText(frame, str(
                    prob), (box[2], box[3]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        except:
            pass
        return frame
       #end
       
    def CNNpreprocessing(self,img):
        img=cv2.resize(img,(24,24));
        img = img.astype('float32')
        img=img/255
        img=np.expand_dims(img,axis=0)    
        return img
    
    def find_eyes_and_crop(self,img,detected_faces):  
        if len(detected_faces)>1:
            face=detected_faces[0] 
        elif len(detected_faces)==0:
            return []
        else:
            [face]=detected_faces
        face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))  
        shape = predictor(img, face_rect)
        shape = face_utils.shape_to_np(shape)

        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        
        """
        img_=img
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(img_, [leftEyeHull], -8, (0, 0, 255), 2)
        cv2.drawContours(img_, [rightEyeHull], -8, (0, 0, 255), 2)
        """
        l=abs(leftEye[3][0]-rightEye[0][0])//2
        distance=abs(leftEye[3][0]-leftEye[0][0])
        c= abs(leftEye[0][1]-leftEye[4][1]+l) if leftEye[4][1] < leftEye[3][1] else  abs(leftEye[0][1]-leftEye[3][1]+l)
        cropleft = img[leftEye[0][1]-c:leftEye[0][1]+c, leftEye[0][0]-distance:leftEye[0][0]+distance+l]      
        distance2=abs(rightEye[3][0]-rightEye[0][0])
        c2= abs(rightEye[0][1]-rightEye[4][1]+l) if rightEye[4][1] < rightEye[3][1] else  abs(rightEye[0][1]-rightEye[3][1]+l)   
        cropright = img[rightEye[0][1]-c2:rightEye[0][1]+c2, rightEye[0][0]-(distance2):rightEye[0][0]+(distance2+l)] 
        return cropright,cropleft
          
    def run(self,capture):  
            
        def shape_to_np(shape, dtype="int"):
           coords = np.zeros((68, 2), dtype=dtype)
           for i in range(0, 68):
             coords[i] = (shape.part(i).x, shape.part(i).y)
           return coords
 
        detector = dlib.get_frontal_face_detector()
        count_yawn = 0
        mouth_output = "GOOD"

        close_counter = blinks = mem_counter= 0
        state = ''
      
        while (True):
            ret,frame=capture.read()
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            if not ret:
                print("Can't receive frame...exiting")
                break
            
            for (i, rect) in enumerate (rects):
                 shape = predictor(gray, rect)
                 shape = face_utils.shape_to_np(shape)
                 Yawn = (abs(shape[49,1]-shape[59,1])+abs(shape[61,1]-shape[67,1])+abs(shape[62,0]-shape[66,0])
                        +abs(shape[63,0]-shape[65,0])+abs(shape[53,0]-shape[55,0]))/abs(shape[48,0]-shape[54,0])
                 
                 if Yawn < 0.8:
                     count_yawn = 0
                     mouth_output = "GOOD"
                 elif Yawn >= 0.8:
                     mouth_output = "Felling Drowsy"
                     count_yawn = count_yawn + 1
           
                 for (x, y) in shape:
                    cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
    
        
            try:    
                boxes, probs, landmarks = self.mtcnn.detect(frame, landmarks=True)               
                right_eye,left_eye=self.find_eyes_and_crop(frame,boxes)              
                prediction=model.predict(self.CNNpreprocessing(right_eye))+model.predict(self.CNNpreprocessing(left_eye))/2.0
                print(prediction)
                         
                if (prediction/2)>0.56 :
                    state = "Open"
                    close_counter = 0
                     
            
                elif (prediction/2)<0.55 :
                    if state == "Drowsy":
                        close_counter = close_counter + 1
                        blinks += 1 
                    if close_counter>=1:
                        close_counter = close_counter +1
                        state = "Drowsy Alarm"
                        sound.play()
                    
                    elif state == "Open":
                        state = "Drowsy"
                        close_counter = 1
                        
                if prediction!=None:     
                    print(state) 

          
            except:
                pass
        
            cv2.putText(frame, "State: {}".format(state), (10, 30),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Count: {}".format(blinks), (10, 50),
			  cv2.FONT_HERSHEY_SIMPLEX,  0.5, (0, 255, 0), 2)
            cv2.putText(frame, "Yawn: {}".format(mouth_output), (10, 70),
			  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame,"Drowsiness Rate: {}".format(count_yawn), (10, 90),
              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
            cv2.imshow('Drowsiness Detection', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                  break
        capture.release()
        cv2.destroyAllWindows()
        del(capture)


mtcnn = MTCNN()
fcd = FaceDetector(mtcnn)
t=threading.Thread(fcd.run(capture)) 
t.daemon=True
t.start()
    
    