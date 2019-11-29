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
import cv2.aruco as aruco
from aruco_lib import *
import copy



########################################################################
## using os to generalise Input-Output
########################################################################
codes_folder_path = os.path.abspath('.')
images_folder_path = os.path.abspath(os.path.join('..', 'Videos'))
generated_folder_path = os.path.abspath(os.path.join('..', 'Generated'))




############################################
## Build your algorithm in this function
## ip_image: is the array of the input image
## imshow helps you view that you have loaded
## the corresponding image
############################################
def process(ip_image):
    ###########################
    '''
    Reference: 1) https://yuzhikov.com/articles/BlurredImagesRestoration1.html
               2) Richard E Woods.
    '''

    list_nps = []

    for i in range(3):
        img = ip_image[:, :, i]
        img = np.float32(img)/255.0
        IMAGE = cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT)
        param = 20
        noise = 10**(-0.1*25)
        sz1 = 65
        kernel = np.ones((1, param), np.float32)
        A = np.float32([[0, -1, 0], [1, 0, 0]])
        sz2 = 32
        A[:,2] = (sz2, sz2) - np.dot(A[:,:2], ((param-1)*0.5, 0))
        point_spread_fun = cv2.warpAffine(kernel, A, (sz1, sz1), flags=cv2.INTER_CUBIC)
        point_spread_fun = point_spread_fun/point_spread_fun.sum()
        point_spread_fun_pad = np.zeros_like(img)
        rows, cols = point_spread_fun.shape
        point_spread_fun_pad[:rows, :cols] = point_spread_fun
        POINT_SPREAD = cv2.dft(point_spread_fun_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows = rows)
        POINTSPREAD2 = (POINT_SPREAD**2).sum(-1)
        INVERSE_POINTSPREAD = POINT_SPREAD / (POINTSPREAD2 + noise)[...,np.newaxis]
        FIN_IMG = cv2.mulSpectrums(IMAGE, INVERSE_POINTSPREAD, 0)
        FINAL_IMAGE = cv2.idft(FIN_IMG, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT )
        FINAL_IMAGE = np.roll(FINAL_IMAGE, -rows//2, 0)
        FINAL_IMAGE = np.roll(FINAL_IMAGE, -cols//2, 1)
        list_nps.append(FINAL_IMAGE)

    final_img = cv2.merge((list_nps[0], list_nps[1], list_nps[2]))
    final_img = final_img*255
    final_img = final_img.astype(np.uint8)
    final_img = cv2.convertScaleAbs(final_img,None,1.9,0)
    detected_marker = detect_Aruco(final_img)
    ID = (list(detected_marker.keys()))[0]
    bot_info = calculate_Robot_State(final_img, detected_marker)
    img = mark_Aruco(final_img, detected_marker)
    cv2.imwrite(os.path.join(os.getcwd() , '..', 'Generated', 'aruco_with_id.png'), img)
    ###########################
    id_list = bot_info[ID]

    return ip_image, id_list


    
####################################################################
## The main program which provides read in input of one image at a
## time to process function in which you will code your generalized
## output computing code
## Do not modify this code!!!
####################################################################
def main(val):
    ################################################################
    ## variable declarations
    ################################################################
    i = 1
    ## reading in video 
    cap = cv2.VideoCapture(images_folder_path+"/"+"ArUco_bot.mp4")
    ## getting the frames per second value of input video
    fps = cap.get(cv2.CAP_PROP_FPS)
    ## getting the frame sequence
    frame_seq = int(val)*fps
    ## setting the video counter to frame sequence
    cap.set(1,frame_seq)
    ## reading in the frame
    ret, frame = cap.read()
    ## verifying frame has content
    print(frame.shape)
    ## display to see if the frame is correct
    cv2.imshow("window", frame)
    cv2.waitKey(0);
    ## calling the algorithm function
    op_image, aruco_info = process(frame)
    ## saving the output in  a list variable
    line = [str(i), "Aruco_bot.jpg" , str(aruco_info[0]), str(aruco_info[3])]
    ## incrementing counter variable
    i+=1
    ## verifying all data
    print(line)
    ## writing to angles.csv in Generated folder without spaces
    with open(generated_folder_path+"/"+'output.csv', 'w') as writeFile:
        print("About to write csv")
        writer = csv.writer(writeFile)
        writer.writerow(line)
    ## closing csv file    
    writeFile.close()



    

############################################################################################
## main function
############################################################################################
if __name__ == '__main__':
    main(input("time value in seconds:"))