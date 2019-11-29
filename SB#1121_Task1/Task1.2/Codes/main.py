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
    ## placeholder image
    sector_image = np.ones(ip_image.shape[:2],np.uint8)*255
    ## check value is white or not
    print(sector_image[0,0])
    img_gray = cv2.cvtColor(ip_image, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img_gray,254,255,cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

    #Check if its 2 or 3.
    rows, cols = ip_image.shape[:2]
    center_x, center_y = rows//2, cols//2
    matrix1 = (contours[-2])[0]
    matrix2 = (contours[-1])[0]

    dist1 = math.sqrt((matrix1[0, 0] - center_x)**2 + (matrix1[0, 1] - center_y)**2)
    dist2 = math.sqrt((matrix2[0, 0] - center_x)**2 + (matrix2[0, 1] - center_y)**2)
    if dist1 > dist2:
        req_matrix = len(contours) - 2
    else:
        req_matrix = len(contours) - 1

    cv2.drawContours(sector_image, contours, req_matrix, color=0, thickness=-1)
    ###########################
    cv2.imshow("window", sector_image)
    cv2.waitKey(0);
    return sector_image




    
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
    ## Reading 1 image at a time from the Images folder
    for image_name in os.listdir(images_folder_path):
        ## verifying name of image
        print(image_name)
        ## reading in image 
        ip_image = cv2.imread(images_folder_path+"/"+image_name)
        ## verifying image has content
        print(ip_image.shape)
        ## passing read in image to process function
        sector_image = process(ip_image)
        ## saving the output in  an image of said name in the Generated folder
        cv2.imwrite(generated_folder_path+"/"+"image_"+str(i)+"_fill_in.png", sector_image)
        i+=1


    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main()
