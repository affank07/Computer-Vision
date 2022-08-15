'''
All of your implementation should be in this file.
'''
'''
This is the only .py file you need to submit. 
'''
'''
    Please do not use cv2.imwrite() and cv2.imshow() in this function.
    If you want to show an image for debugging, please use show_image() function in helper.py.
    Please do not save any intermediate files in your final submission.
'''

from helper import show_image

import cv2
import numpy as np
import os
import sys

import face_recognition

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''


def detect_faces(input_path: str) -> dict:
    result_list = []
    '''
    Your implementation.
    '''
    images = []
    images_name=[]
    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path,filename))
        if img is not None:
            images.append(img)
            images_name.append(filename)


    #*********************************************** Haar Cascade *************************************************************


    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # for i in range(len(images)):
    #     gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    #     faces = face_cascade.detectMultiScale(gray, 1.2, 4)
    #     for (x,y,w,h) in faces:
    #             result_list.append({"iname": images_name[i], "bbox": [int(x), int(y), int(w), int(h)]})
        

    #********************************************* End of Haar Cascade *********************************************************






    #*************************************************** DNN ******************************************************************


    model = "opencv_face_detector_uint8.pb"
    config = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(model, config)

    for i in range(len(images)):
            h1, w1 = images[i].shape[:2]
            blob = cv2.dnn.blobFromImage(images[i], 0.8, (227,227))
            net.setInput(blob)
            op = net.forward()
            for j in range(op.shape[2]):
                confidence = op[0, 0, j, 2]
                if confidence > 0.45:
                    face = op[0, 0, j, 3:7] * np.array([w1, h1, w1, h1])
                    x = int(face[0])
                    y = int(face[1])
                    w = (int(face[2]) - int(face[0]))
                    h = (int(face[3]) - int(face[1]))
                    result_list.append({"iname": images_name[i], "bbox": [int(x), int(y), int(w), int(h)]})
                    

    #************************************************ End of DNN **************************************************************





    #***************************************************** HOG *****************************************************************


        #************************************************ Method 1 *************************************************************

    # for i in range(len(images)):
    #     img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
    #     face_locations = face_recognition.face_locations(img)
    #     for (top,right,bottom,left) in face_locations:
    #         x=left
    #         y=top
    #         w=right-x
    #         h=bottom-y
    #         result_list.append({"iname": images_name[i], "bbox": [int(x), int(y), int(w), int(h)]})
    

        #*********************************************** End of Method 1 ********************************************************


        #************************************************* Method 2 *************************************************************


    # import dlib

    # rgb=[]
    # for i in range(len(images)):
    #     img=cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
    #     rgb.append(img)
    
    # detector = dlib.get_frontal_face_detector()

    # for i in range(len(rgb)):
    #     rects = detector(rgb[i])
    #     for d in rects:
    #         startX = d.left()
    #         startY = d.top()
    #         endX = d.right()
    #         endY = d.bottom()
    #         startX = max(0, startX)
    #         startY = max(0, startY)
    #         endX = min(endX, rgb[i].shape[1])
    #         endY = min(endY, rgb[i].shape[0])
    #         w = endX - startX
    #         h = endY - startY
    #         x=startX
    #         y=startY

    #         result_list.append({"iname": images_name[i], "bbox": [int(x), int(y), int(w), int(h)]})


        #******************************************** End of Method 2 *********************************************************


    #************************************************** End of HOG ************************************************************


    return result_list


'''
K: number of clusters
'''
def cluster_faces(input_path: str, K: int) -> dict:
    result_list = []
    crop_dim=[]
    '''
    Your implementation.
    '''
    images = []
    images_name=[]
    for filename in os.listdir(input_path):
        img = cv2.imread(os.path.join(input_path,filename))
        if img is not None:
            images.append(img)
            images_name.append(filename)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for i in range(len(images)):
        gray = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 4)
        for (x,y,w,h) in faces:
                crop_dim.append([int(x), int(y), int(w), int(h)])
    #print('faces detected:', len(crop_dim))
    
    crop_faces=[]
    for i in range(len(images)):
        #cropped_image = images[i][crop_dim[i][0]:crop_dim[i][0]+crop_dim[i][2], crop_dim[i][1]:crop_dim[i][1]+crop_dim[i][3]]
        cropped_image = images[i][crop_dim[i][1]:crop_dim[i][1]+crop_dim[i][3], crop_dim[i][0]:crop_dim[i][0]+crop_dim[i][2]]
        crop_faces.append(cropped_image)

    crop_vector=[]
    for i in range(len(images)):
        boxes=[(crop_dim[i][1],crop_dim[i][0]+crop_dim[i][2],crop_dim[i][1]+crop_dim[i][3],crop_dim[i][0])]
        vector=face_recognition.face_encodings(images[i], boxes)
        crop_vector.append(vector[0])
    #print(len(crop_vector))

    from sklearn.cluster import KMeans
    K=int(K)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(crop_vector)
    # print(kmeans.labels_)

    # labels_count=[]
    # for i in np.unique(kmeans.labels_):
    #     count=0
    #     for j in kmeans.labels_:
    #         if i==j:
    #             count+=1
    #     labels_count.append(count)
    
    # print(labels_count)


    groups = {}
    groups_name={}

    for img, cluster in zip(crop_faces,kmeans.labels_):
        if cluster not in groups.keys():
            groups[cluster] = []
            groups[cluster].append(img)
        else:
            groups[cluster].append(img)

    for name, cluster in zip(images_name,kmeans.labels_):
        if cluster not in groups_name.keys():
            groups_name[cluster] = []
            groups_name[cluster].append(name)
        else:
            groups_name[cluster].append(name)
    
    for i in range(len(groups_name)):
        result_list.append({"cluster_no": i, "elements": groups_name[i]})

    #*********************************************** Plotting clusters *******************************************************

    # import matplotlib.pyplot as plt
    
    
    # for i in range(len(groups)):
    #     for j in range(len(groups[i])):
    #         plt.subplot(len(groups), len(groups[i]), j+1)
    #         plt.xticks([])
    #         plt.yticks([])
    #         plt.grid(False)
    #         plt.imshow(groups[i][j])
    #     plt.show()

    return result_list


'''
If you want to write your implementation in multiple functions, you can write them here. 
But remember the above 2 functions are the only functions that will be called by FaceCluster.py and FaceDetector.py.
'''

"""
Your implementation of other functions (if needed).
"""
