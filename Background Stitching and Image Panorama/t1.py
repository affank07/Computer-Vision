#Only add your code inside the function (including newly improted packages)
# You can design a new function and call the new function in the given functions. 
# Not following the project guidelines will result in a 10% reduction in grades

import cv2
import numpy as np
import matplotlib.pyplot as plt


def stitch_background(img1, img2, savepath=''):
    "The output image should be saved in the savepath."
    "Do NOT modify the code provided."
    img1=img1
    img2=img2
    img2 = cv2.copyMakeBorder(img2, 200, 200, 200, 200, cv2.BORDER_CONSTANT, (0,0,0))
    sift=cv2.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)
    #print(len(kp1))
    #print(desc1.shape)
    #print(len(kp2))
    #print(desc2.shape)

    #img2 = cv2.copyMakeBorder(img2, 200, 200, 200, 200, cv2.BORDER_CONSTANT, (0,0,0))

    match1=[]
    match2=[]

    for i1 in range(desc1.shape[0]):
        dis_list=[]
        for i2 in range(desc2.shape[0]):
            euc_dis=np.linalg.norm(desc1[i1]-desc2[i2])
            dis_list.append(euc_dis)
        min_desc2=min(dis_list)
        index2=dis_list.index(min_desc2)
        dis_list.sort()
        thresh=0.7
        #ratio=dis_list[0]/dis_list[1]
        if dis_list[0]<thresh*dis_list[1]:
            #print(i1)
            match1.append([i1,index2])
    #print(len(match1))
    #print(match1[0])
    #print(desc1[6])
    #print(desc2[229])

    pts_1=np.float32([kp1[m[0]].pt for m in match1]).reshape(-1,1,2)
    #print(pts_1)
    pts_2=np.float32([kp2[m[1]].pt for m in match1]).reshape(-1,1,2)
    #print(pts_2)

    matrix, mask = cv2.findHomography(pts_1, pts_2, cv2.RANSAC, 5.0)
    #print(matrix)

    dst = cv2.warpPerspective(img1,matrix,((img2.shape[1] + img1.shape[1]), img1.shape[0]+img2.shape[0])) #wraped image
    #dst1=cv2.warpPerspective(img1,matrix,((img2.shape[1] + img1.shape[1]), img2.shape[0]))
    #dst[0:img1.shape[0], 0:img1.shape[1]] = img1
    # dst[0:img2.shape[0], 0:img2.shape[1]] = img2
    # dst[0:img1.shape[0], 0:img1.shape[1]] = img1
    # cv2.imwrite('output.jpg',dst)
    # plt.imshow(dst)
    # plt.show()
    back=np.zeros(dst.shape)
    back[0:img2.shape[0], 0:img2.shape[1]]=img2
    #plt.imshow(back)
    #plt.show()
    #cv2.imwrite('output.jpg',back)
    #print(dst.shape)
    #print(back.shape)

    for i in range(back.shape[0]):
        for j in range(back.shape[1]):
            for k in range(back.shape[2]):
                if back[i,j,k]>dst[i,j,k]:
                    back[i,j,k]=back[i,j,k]
                elif dst[i,j,k]>back[i,j,k]:
                    back[i,j,k]=dst[i,j,k]
    #cv2.imwrite('output.jpg',back)
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

    res=back[xmin:xmax,ymin:ymax]
    #print(res.shape)
    cv2.imwrite(savepath,res)


    return
if __name__ == "__main__":
    img1 = cv2.imread('./images/t1_1.png')
    img2 = cv2.imread('./images/t1_2.png')
    savepath = 'task1.png'
    stitch_background(img1, img2, savepath=savepath)

