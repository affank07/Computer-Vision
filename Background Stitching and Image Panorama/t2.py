# 1. Only add your code inside the function (including newly improted packages). 
#  You can design a new function and call the new function in the given functions. 
# 2. For bonus: Give your own picturs. If you have N pictures, name your pictures such as ["t3_1.png", "t3_2.png", ..., "t3_N.png"], and put them inside the folder "images".
# 3. Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json


def stitch(imgmark, N=4, savepath=''): #For bonus: change your input(N=*) here as default if the number of your input pictures is not 4.
    "The output image should be saved in the savepath."
    "The intermediate overlap relation should be returned as NxN a one-hot(only contains 0 or 1) array."
    "Do NOT modify the code provided."
    imgpath = [f'./images/{imgmark}_{n}.png' for n in range(1,N+1)]
    imgs = []
    for ipath in imgpath:
        img = cv2.imread(ipath)
        imgs.append(img)
    "Start you code here"
    
    sift=cv2.xfeatures2d.SIFT_create()  

    #Overlapping Matrix
    overlap_arr1=np.zeros([N,N])
    match_val=np.zeros([N,N])
    for i in range(len(imgs)):
        for j in range(len(imgs)):


            img1=imgs[i]
            img2=imgs[j]
            kp1, desc1 = sift.detectAndCompute(img1, None)
            kp2, desc2 = sift.detectAndCompute(img2, None)
            #print(len(kp1))
            #print(desc1.shape)
            #print(len(kp2))
            #print(desc2.shape)

            #img2 = cv2.copyMakeBorder(img2, 200, 200, 200, 200, cv2.BORDER_CONSTANT, (0,0,0))

            match1=[]

            for i1 in range(desc1.shape[0]):
                dis_list=[]
                for i2 in range(desc2.shape[0]):
                    euc_dis=np.linalg.norm(desc1[i1]-desc2[i2])
                    dis_list.append(euc_dis)
                min_desc2=min(dis_list)
                index2=dis_list.index(min_desc2)
                dis_list.sort()
                thresh=0.8
                #ratio=dis_list[0]/dis_list[1]
                if dis_list[0]<thresh*dis_list[1]:
                    #print(i1)
                    match1.append([i1,index2])
            #print(len(match1))
            if len(kp1)<len(kp2):
                kp=kp1
            elif len(kp2)<len(kp1):
                kp=kp2
            elif len(kp1)==len(kp2):
                kp=kp1
            match_val[i][j]=len(match1)
            if len(match1)>0.2*len(kp):
                overlap_arr1[i,j]=1
                
    #print(overlap_arr1)
    overlap_arr=overlap_arr1

    #Removing if an image doesn't match with any other image 
    for i in range(overlap_arr1.shape[0]):
        if np.sum(overlap_arr1[i])>1.0:
            pass
        else:
            imgs.remove(imgs[i])

    #Order of images
    matches_others=[]
    for i in range(overlap_arr1.shape[0]):
        matches_others.append(np.sum(overlap_arr1[i]))

    matches_max = np.argmax(matches_others)
    matches_count = -match_val[matches_max]
    
    imgs_sorted = np.argsort(matches_count)
    #print(imgs_sorted)
    
    new_imgs=[]
    for i in imgs_sorted:
        new_imgs.append(imgs[i])
    imgs=new_imgs

    
    #print(len(imgs))
 
    n=0
    #Stiching the images
    for i in range(len(imgs)-1):
        if n==0:
            img1=imgs[i]
            img1=cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            img2=imgs[i+1]
            img2=cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        elif n!=0:
            img1=imgs[i+1]
            img1=cv2.normalize(img1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')   
            img2=back
            img2=cv2.normalize(img2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            
        n+=1

        img1 = cv2.copyMakeBorder(img1, 500, 500, 500, 500, cv2.BORDER_CONSTANT, (0,0,0))
        img2 = cv2.copyMakeBorder(img2, 500, 500, 500, 500, cv2.BORDER_CONSTANT, (0,0,0))
        
        
        kp1, desc1 = sift.detectAndCompute(img1, None)
        kp2, desc2 = sift.detectAndCompute(img2, None)

        match=[]
        
        for i1 in range(desc1.shape[0]):
            dis_list=[]
            for i2 in range(desc2.shape[0]):
                euc_dis=np.linalg.norm(desc1[i1]-desc2[i2])
                dis_list.append(euc_dis)
            min_desc2=min(dis_list)
            index2=dis_list.index(min_desc2)
            dis_list.sort()
            thresh=0.8
            #ratio=dis_list[0]/dis_list[1]
            if dis_list[0]<thresh*dis_list[1]:
                #print(i1)
                match.append([i1,index2])

        pts_1=np.float32([kp1[m[0]].pt for m in match]).reshape(-1,1,2)
        #print(pts_1)
        pts_2=np.float32([kp2[m[1]].pt for m in match]).reshape(-1,1,2)
        #print(pts_2)

        matrix, mask = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, 5.0)
        #print(matrix)
        
        dst = cv2.warpPerspective(img1,matrix,((img2.shape[1] + img1.shape[1]), img1.shape[0]+img2.shape[0]))
        back=np.zeros(dst.shape)
        back[0:img2.shape[0], 0:img2.shape[1]]=img2
        for i in range(back.shape[0]):
            for j in range(back.shape[1]):
                for k in range(back.shape[2]):
                    if back[i,j,k]>dst[i,j,k]:
                        back[i,j,k]=back[i,j,k]
                    elif dst[i,j,k]>back[i,j,k]:
                        back[i,j,k]=dst[i,j,k]

        x_pnts=[]
        y_pnts=[]
        for x in range(back.shape[0]):
            for y in range(back.shape[1]):
                for k in range(back.shape[2]):
                    if back[x,y,k]>0:
                        x_pnts.append(x)
                        y_pnts.append(y)

        xmax=max(x_pnts)
        xmin=min(x_pnts)
        ymax=max(y_pnts)
        ymin=min(y_pnts)

        back=back[xmin:xmax,ymin:ymax] 

    cv2.imwrite(savepath,back)

    #print('Task Done!!!')
    return overlap_arr


if __name__ == "__main__":
    #task2
    overlap_arr = stitch('t2', N=4, savepath='task2.png')
    with open('t2_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr.tolist(), outfile)
    #bonus
    overlap_arr2 = stitch('t3', savepath='task3.png')
    with open('t3_overlap.txt', 'w') as outfile:
        json.dump(overlap_arr2.tolist(), outfile)
