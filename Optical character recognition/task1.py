"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provided.
Please follow the guidelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np

def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: width of the detected character.
        h: height of the detected character.
        name: name of character provided or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments

    chars,charname=enrollment(characters)
    
    bbox,cropchars=detection(test_img)
    
    a = recognition(chars,charname,bbox,cropchars)

    return(a)

    #raise NotImplementedError

def enrollment(chars):
    """ Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    
    count=0
    for i in chars:
        b,chars[count][1]=cv2.threshold(chars[count][1],150,255,cv2.THRESH_BINARY_INV)
        chars[count][1]=cv2.GaussianBlur(i[1],(5,5),cv2.BORDER_DEFAULT)
        chars[count][1]=cv2.resize(chars[count][1], (25,25) )
        #print(chars[count][1].shape)
        count+=1
        
    #print(chars)
    
    char=[]
    for i in range(count):
        char.append(chars[i][1])
        points=np.where(char[i]==0)
        xmin=np.min(points[0])
        xmax=np.max(points[0])
        ymin=np.min(points[1])
        ymax=np.max(points[1])
        char[i]=char[i][xmin:xmax,ymin:ymax]
        char[i]=cv2.resize(char[i], (30,30) )
        #cv2.imshow('image', char[i])
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    charsname=[]
    for i in range(count):
        charsname.append(chars[i][0])
    #print(charsname)
    #cv2.imshow('image', char[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()


    

    return(char,charsname)
    
    
    # TODO: Step 1 : Your Enrollment code should go here.
    #print(keypoint[0])
    #print(descriptor[0])
    #raise NotImplementedError

def detection(test_img):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 2 : Your Detection code should go here.
    #raise NotImplementedError
    bbox,cropchars=CCL(test_img)
    #print(bbox[0])
    #cv2.imshow('B',cropchars[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    return(bbox,cropchars)

    

def recognition(chars,charsname,bbox,cropchars):
    """ 
    Args:
        You are free to decide the input arguments.
    Returns:
    You are free to decide the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    #cv2.imshow('2',chars[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(charsname)
    #print(bbox[0])
    #cv2.imshow('B',cropchars[0])
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #print(chars[0].shape)
    #print(cropchars[0].shape)
    normalize=100000
    results=[]
    

    for i in range(len(cropchars)):   #count
        ssds=[]
        least=70
        least2=0
        correct=False
        
        for j in range(len(chars)):    #datainchar
            ssdval=np.sum((cropchars[i]-chars[j])**2)
            ssdval=(float(ssdval)/normalize)
            ssds.append(ssdval)
            
            
            
            if min(ssds)<least:
                least=min(ssds)
                correct=True
                least2=j
        if correct==False:
            results.append({'bbox':bbox[i],'name':'UNKNOWN'})
        elif correct==True:
            results.append({'bbox':bbox[i],'name':charsname[least2]})
        
    return(results)
        




    
    #raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = coordinates
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    print(all_character_imgs)

    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)
    
    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)

def CCL(test_img):
    count=0
    labels=[]
    testimg = test_img
    #print(testimg.shape)
    tempimg = np.zeros((testimg.shape[0],testimg.shape[1]))


    for i in range(testimg.shape[0]):
        for j in range(testimg.shape[1]):
            
            if (testimg[i,j]>225):
                    tempimg[i,j] = 0
            else:
                tempimg[i,j] = 255

    #print(tempimg.shape)
    #cv2.imshow('Grayscaled',tempimg)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    labelmatrix=np.zeros((tempimg.shape[0],tempimg.shape[1]))

    for i in range(labelmatrix.shape[0]):
        for j in range(labelmatrix.shape[1]):
            if tempimg[i,j]!=0 and tempimg[i,j-1]==0 and tempimg[i-1,j]==0:
                count+=1
                labels.append(0)
                labelmatrix[i,j]=count
                
            elif tempimg[i,j]!=0 and tempimg[i,j-1]==0 and tempimg[i-1,j]!=0:
                labelmatrix[i,j]=labelmatrix[i-1,j]
                
            elif tempimg[i,j]!=0 and tempimg[i,j-1]!=0 and tempimg[i-1,j]==0:
                labelmatrix[i,j]= labelmatrix[i,j-1]
                
            elif tempimg[i,j]!=0 and tempimg[i,j-1]!=0 and tempimg[i-1,j]!=0:
                labelmatrix[i,j]=min(labelmatrix[i,j-1],labelmatrix[i-1,j])
                if labelmatrix[i-1,j]!=labelmatrix[i,j-1]:            
                    if labelmatrix[i,j-1]>labelmatrix[i-1,j]:
                        labels[int(labelmatrix[i,j-1]-1)]=labelmatrix[i-1,j]
                    elif labelmatrix[i,j-1]<labelmatrix[i-1,j]:
                        labels[int(labelmatrix[i-1,j]-1)]=labelmatrix[i,j-1]
        
            elif tempimg[i,j]==0:
                labelmatrix[i,j]=0
            
                
                        
    #print(len(np.unique(labels)))




    for i in range(labelmatrix.shape[0]):
        for j in range(labelmatrix.shape[1]):

            if labelmatrix[i,j]!=0:
                
                if labelmatrix[i,j-1]==0 and labelmatrix[i-1,j]==0:
                    labelmatrix[i,j]=labelmatrix[i,j]

                if labelmatrix[i,j-1]!=0 and labelmatrix[i-1,j]==0:
                    samelabel=min(labelmatrix[i,j],labelmatrix[i,j-1])
                    labelmatrix[i,j]=samelabel
                    labelmatrix[i,j-1]=samelabel
                    

                if labelmatrix[i,j-1]==0 and labelmatrix[i-1,j]!=0:
                    samelabel=min(labelmatrix[i,j],labelmatrix[i-1,j])
                    labelmatrix[i,j]=samelabel
                    labelmatrix[i-1,j]=samelabel
                    

                if labelmatrix[i,j-1]!=0 and labelmatrix[i-1,j]!=0: 
                    
                    if labelmatrix[i,j-1]!=labelmatrix[i-1,j]:
                        samelabel=min(labelmatrix[i,j-1],labelmatrix[i-1,j],labelmatrix[i,j])
                        labelmatrix[i,j]=samelabel
                        labelmatrix[i,j-1]=samelabel
                        labelmatrix[i-1,j]=samelabel
                        

                    if labelmatrix[i,j-1]==labelmatrix[i-1,j]:
                        samelabel=min(labelmatrix[i,j-1],labelmatrix[i-1,j],labelmatrix[i,j])
                        labelmatrix[i,j]=samelabel
                        labelmatrix[i,j-1]=samelabel
                        labelmatrix[i-1,j]=samelabel
                        
            elif labelmatrix[i,j]==0:
                    if labelmatrix[i,j-1]!=0 and labelmatrix[i-1,j]!=0:
                        if labelmatrix[i,j-1]!=labelmatrix[i-1,j]:
                            samelabel=min(labelmatrix[i,j-1],labelmatrix[i-1,j])
                            labelmatrix[i,j-1]=samelabel
                            labelmatrix[i-1,j]=samelabel


    #print(len(np.unique(labelmatrix)))

                
    newlabels=np.zeros(len(labels))
    label=1
    done=False

    while label < len(labels):
        dl=label
        done=False
        while labels[int(dl-1)]!=0:
            dl=labels[int(dl-1)]
            done = True        
        if done:
            newlabels[int(label-1)]=dl
        label+=1
        done = False
                
    i=0              
    while i <labelmatrix.shape[0]:
        j=0
        while j<labelmatrix.shape[1]:
            if labelmatrix[i,j]!=0 and newlabels[int(labelmatrix[i,j])-1]!=0:
                labelmatrix[i,j]=newlabels[int(labelmatrix[i,j])-1]
            j+=1
        i+=1                
    #print(len(np.unique(labelmatrix)))

    charslabel = np.unique(labelmatrix)
    print(len(charslabel))
    charslabel=np.delete(charslabel,0)
    #print(charslabel))

    charlabelpos=[]
    pos=[]

    for p in charslabel:
        for i in range(labelmatrix.shape[0]):
            for j in range(labelmatrix.shape[1]):
                if labelmatrix[i,j]==p:
                    pos.append([i,j])
        charlabelpos.append(pos)
        pos=[]

    #print(charlabelpos[0])

    character_list = []
    count=0
    xpos=[]
    ypos=[]
    results=[]
    points=[]
    bbox=[]

    for labelpos in charlabelpos:
        for point in labelpos:
            points.append(point)
            xpos.append(point[0])
            ypos.append(point[1])
        xmin=min(xpos)
        xmax=max(xpos)
        ymin=min(ypos)
        ymax=max(ypos)
        widht=xmax-xmin
        height=ymax-ymin
        crop= labelmatrix[xmin:xmax,ymin:ymax]
        #print(a)
        crop = character_list.append(crop)
        
        
        dic={'bbox':[points[0][1],points[0][0],height,widht],
            'name': 'UNKNOWN'}
        results.append(dic)
        bbox.append([points[0][1],points[0][0],height,widht])
        
        xpos=[]
        ypos=[]
        points=[]
        
    count=0
    for i in character_list:
        
        a,character_list[count]=cv2.threshold(character_list[count],0.5,255,cv2.THRESH_BINARY_INV)
        character_list[count]=cv2.GaussianBlur(i,(5,5),cv2.BORDER_DEFAULT)
        character_list[count]=cv2.resize(character_list[count],(30,30))
        
        count+=1
    

    #print(results)
    return(bbox,character_list)   




if __name__ == "__main__":
    main()

