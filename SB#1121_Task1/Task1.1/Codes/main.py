###############################################################################
## Author: Team Supply Bot
## Edition: eYRC 2019-20
## Instructions: Do Not modify the basic skeletal structure of given APIs!!!
###############################################################################


######################
## Essential libraries
######################
import cv2
import numpy as np
import os
import math
import csv




########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Images'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))




############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################
def process(ip_image):
    ###########################
    ## Your Code goes here
    image_gray = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
    matrix1 = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,1,20, param1=50,param2=30,minRadius=0,maxRadius=20)

    for lists in matrix1[0, :]:
        if (ip_image[int(lists[1]), int(lists[0])] == [255, 255, 255]).all():
            white = lists
            continue
        elif (ip_image[int(lists[1]), int(lists[0])] == [0, 255, 0]).all():
            green = lists
            continue
        elif (ip_image[int(lists[1]), int(lists[0])] == [0, 0, 255]).all():
            red = lists
            continue

    dist1 = math.sqrt((red[0] - white[0])**2 + (red[1] - white[1])**2) #a
    dist2 = math.sqrt((green[0] - white[0])**2 + (green[1] - white[1])**2) #b
    dist3 = math.sqrt((red[0] - green[0])**2 + (red[1] - green[1])**2) #c
    angle = round((math.acos((dist1**2 + dist2**2 - dist3**2)/(2*dist1*dist2)))*(180/math.pi), 2)
    
    ## Your Code goes here
    ###########################
    cv2.imshow("window", ip_image)
    cv2.waitKey(0);
    return angle






    
####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main():
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    line = []
    ## Reading 1 image at a time from the Images folder
    for image_name in os.listdir(images_folder_path):
        ## verifying name of image
        print(image_name)
        ## reading in image 
        ip_image = cv2.imread(images_folder_path+"/"+image_name)
        ## verifying image has content
        print(ip_image.shape)
        ## passing read in image to process function
        A = process(ip_image)
        ## saving the output in  a list variable
        line.append([str(i), image_name , str(A)])
        ## incrementing counter variable
        i+=1
    ## verifying all data
    print(line)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path+"/"+'angles.csv', 'w', newline='') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(line)
    ## closing csv file    
    writeFile.close()



    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()
