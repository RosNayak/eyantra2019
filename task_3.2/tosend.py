import cv2
import numpy as np
import math

#Reading the image.
img = cv2.imread('IMAGE.jpg', 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Canny edge detection.
edges = cv2.Canny(img_gray, 60, 100, 3)

#Algorithm to crop the image and remove eyantra logo from the image.
contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
new = np.zeros(img_gray.shape, np.uint8)
new = cv2.drawContours(new, contours, -1, (255,255,255), -1)

kernel = np.ones((15, 15), np.uint8)
opening = cv2.morphologyEx(new, cv2.MORPH_OPEN, kernel)

contours, hierarchy = cv2.findContours(opening,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
print(len(contours))
max_area = 0
index = 0
for contour in contours:
	area = cv2.contourArea(contour)

	#if the area is more than the max area.
	if area > max_area:
		max_area = area
		index = contours.index(contour)

new_img = np.zeros(img_gray.shape, np.uint8)
new_img = cv2.drawContours(new_img, contours, index, (255,255,255), -1)
cv2.imshow('image', new_img)
cv2.waitKey(0)

res = cv2.bitwise_and(img,img,mask = new_img)

#Canny edge detection.
contrast = 100
cst = np.int16(res)
cst = cst*(contrast/127 + 1) - contrast + 25
cst = np.clip(cst, 0, 255)
cst = np.uint8(cst)
cv2.imshow('image', cst)
cv2.waitKey(0)
gray_res = cv2.cvtColor(cst, cv2.COLOR_BGR2GRAY)
gray_res = cv2.medianBlur(gray_res, 3)
edges = cv2.Canny(gray_res, 50, 100, 5)

#Detect the contours.
contours1, hierarchy = cv2.findContours(edges,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

#Remove open contours.
closed_cnt = []
for cnt in contours1:
	if cv2.contourArea(cnt) > cv2.arcLength(cnt, True):
		closed_cnt.append(cnt)

#Remove the duplicate contours.
req_cnt = []
count = 0
for cnt in closed_cnt:
	if count%2 == 0:
		req_cnt.append(cnt)
	count += 1

#Calculate the areas of the contours.
areas = {}
key = 0
for cnt in req_cnt:
	area = cv2.contourArea(cnt)
	areas[str(key)] = area
	key += 1

#Sort the areas dictionary.
sorted_dict = sorted(areas.items(), key = lambda kv:(kv[1], kv[0]))

#Take a black image and draw the region of interest.
new_img = np.zeros(img_gray.shape, np.uint8)
for key, value in sorted_dict[0:3]:
	new_img = cv2.drawContours(new_img, req_cnt, int(key), (255,255,255), -1)

#Capture the region of interest from img.
res_again = cv2.bitwise_and(img,img,mask = new_img)

#Gray it and detect the circles.
res_again_gray = cv2.cvtColor(res_again, cv2.COLOR_BGR2GRAY)
res_again_gray = cv2.medianBlur(res_again_gray, 5) #Hough circles works better with blurred images.
circles1 = cv2.HoughCircles(res_again_gray,cv2.HOUGH_GRADIENT,1.55,50, param1=20,param2=10,minRadius=0,maxRadius=15)

#Draw the circles on img.
for circle in circles1[0, :]:
	cv2.circle(img, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)

#Calculate the distances.
circles1 = circles1[0]
if circles1[0,1] == circles1[1,1] or circles1[1,1] == circles1[2,1] or circles1[0,1] == circles1[2,1]:
	sorted_array = circles1[circles1[:,0].argsort(kind='mergesort')]
	dist1 = math.sqrt((sorted_array[0, 0] - sorted_array[1, 0])**2 + (sorted_array[0, 1] - sorted_array[1, 1])**2)
	dist2 = math.sqrt((sorted_array[2, 0] - sorted_array[1, 0])**2 + (sorted_array[2, 1] - sorted_array[1, 1])**2)
	dist3 = math.sqrt((sorted_array[0, 0] - sorted_array[2, 0])**2 + (sorted_array[0, 1] - sorted_array[2, 1])**2)
else:
	sorted_array = circles1[circles1[:,1].argsort(kind='mergesort')]
	dist1 = math.sqrt((sorted_array[0, 0] - sorted_array[1, 0])**2 + (sorted_array[0, 1] - sorted_array[1, 1])**2)
	dist2 = math.sqrt((sorted_array[2, 0] - sorted_array[1, 0])**2 + (sorted_array[2, 1] - sorted_array[1, 1])**2)
	dist3 = math.sqrt((sorted_array[0, 0] - sorted_array[2, 0])**2 + (sorted_array[0, 1] - sorted_array[2, 1])**2)

#Detect the angle.
angle = round((math.acos((dist1**2 + dist2**2 - dist3**2)/(2*dist1*dist2)))*(180/math.pi), 2)

#Write the angle detected on the image and then write the image to the file.
image = cv2.putText(img, 'Angle: '+str(angle), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
cv2.imwrite('RESULT.jpg', image)

#Thats it!! We are done. You have your circles:):):):).
