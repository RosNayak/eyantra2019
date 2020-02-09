import cv2
import numpy as np
import math
import cv2.aruco as aruco
import serial
import csv


#CROPPING THE IMAGE TO GET RID OF THE EYANTRA LOGO.
cap = cv2.VideoCapture(1)
fps = cap.get(cv2.CAP_PROP_FPS)
cap.set(3, 640)
cap.set(4, 480)
ret, img = cap.read()
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(img_gray, 60, 100, 3)
contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
mask = np.zeros(img_gray.shape, np.uint8)
mask = cv2.drawContours(mask, contours, -1, (255,255,255), -1)
kernel = np.ones((15, 15), np.uint8)
extract_ROI = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
contours, hierarchy = cv2.findContours(extract_ROI, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_area = 0
req_cnt = []
for contour in contours:
	area = cv2.contourArea(contour)

	if area > max_area:
		max_area = area
		req_cnt = contour

mask2 = np.zeros(img_gray.shape, np.uint8)
mask2 = cv2.drawContours(mask2, [req_cnt], 0, (255,255,255), -1)
res = cv2.bitwise_and(img,img,mask = mask2)

#CODE TO DETECT THE COINS ON THE ARENA AND PRINT THE NUMBER OF COINS ON THE ARENA.
contrast = 100
cst = np.int16(res)
cst = cst*(contrast/127 + 1) - contrast + 25
cst = np.clip(cst, 0, 255)
cst = np.uint8(cst)

gray_cst = cv2.cvtColor(cst, cv2.COLOR_BGR2GRAY)
gray_cst = cv2.medianBlur(gray_cst, 3)
edges2 = cv2.Canny(gray_cst, 50, 100, 5)

contours2, hierarchy = cv2.findContours(edges2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

closed_cnt = []
for cnt in contours2:
	if cv2.contourArea(cnt) > cv2.arcLength(cnt, True):
		closed_cnt.append(cnt)

req_cnt = []
count = 0
for cnt in closed_cnt:
	if count%2 == 0:
		req_cnt.append(cnt)
	count += 1

areas = {}
key = 0
for cnt in req_cnt:
	area = cv2.contourArea(cnt)
	areas[str(key)] = area
	key += 1

sorted_dict = sorted(areas.items(), key = lambda kv:(kv[1], kv[0]))

mask3 = np.zeros(img_gray.shape, np.uint8)
for key, value in sorted_dict[0:4]:
	mask3 = cv2.drawContours(mask3, req_cnt, int(key), (255,255,255), -1)

res2 = cv2.bitwise_and(img,img,mask = mask3)

res2_gray = cv2.cvtColor(res2, cv2.COLOR_BGR2GRAY)
res2_gray = cv2.medianBlur(res2_gray, 5)
circles = cv2.HoughCircles(res2_gray,cv2.HOUGH_GRADIENT,1.55,50, param1=20,param2=10,minRadius=0,maxRadius=15)
print('Number of coins detected:', len(circles[0]))


#CODE TO DETECT THE CAPITAL STARTS FROM HERE.
retake = img
retake = cv2.resize(retake, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
gray = cv2.cvtColor(retake, cv2.COLOR_BGR2GRAY)
aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_50)
parameters = aruco.DetectorParameters_create()
corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
if corners != []:
	corner_points = corners[0][0]
	for i in range(len(corner_points)):
		corner_points[i, 0] = corner_points[i, 0]*(0.5)
		corner_points[i, 1] = corner_points[i, 1]*(0.5)
retake = cv2.resize(retake, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
capital_coordinates = corner_points.sum(axis=0)//4


#CHANGE CONTRAST OR TRESHOLD VALUE IF REQUIRED THING IS NOT DETECTED PROPERLY.
contrast = 100
cst2 = np.int16(res3)
cst2 = cst2*(contrast/127 + 1) - contrast + 100
cst2 = np.clip(cst2, 0, 255)
cst2 = np.uint8(cst2)

gray_cst2 = cv2.cvtColor(cst2, cv2.COLOR_BGR2GRAY)

ret, tresh = cv2.threshold(gray_cst2, 180, 255, cv2.THRESH_BINARY)

cities_and_debris = cv2.morphologyEx(tresh, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

contours, hierarchy = cv2.findContours(cities_and_debris,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

areas = {}
key = 0
for cnt in contours:
	area = cv2.contourArea(cnt)
	areas[str(key)] = area
	key += 1

sorted_dict = sorted(areas.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

mask4 = np.zeros(img_gray.shape, np.uint8)

for key, value in sorted_dict[1:]: 
	mask4 = cv2.drawContours(mask4, contours, int(key), (255,255,255), -1)

contours, hierarchy = cv2.findContours(mask4,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
centers_of_contours = []
for cnt in contours:
	((x, y), radius) = cv2.minEnclosingCircle(cnt)
	centers_of_contours.append([int(x), int(y)])

distance_dict = {}
for centers in centers_of_contours:
	distance = math.sqrt((centers[0] - capital_coordinates[0])**2 + (centers[1] - capital_coordinates[1])**2)
	distance_dict[int(distance)] = centers

distance_keys = distance_dict.keys()
sorted_keys = sorted(distance_keys, reverse=True)

centers_of_cities = []
for i in range(0, 8):
	centers_of_cities.append(distance_dict[sorted_keys[i]])

centers_of_cities.append([int(capital_coordinates[0]), int(capital_coordinates[1])])

#CODE TO NUMBER THE CITIES AND DETECT THE NODE NUMBER OF THE CITY.
centers_of_cities = sorted(centers_of_cities, key=lambda x: x[1], reverse=True)

if centers_of_cities[1][0] > centers_of_cities[2][0]:
	temp = centers_of_cities[1]
	centers_of_cities[1] = centers_of_cities[2]
	centers_of_cities[2] = temp

for i in range(1, len(centers_of_cities)-1):
	min_distance = math.sqrt((centers_of_cities[i+1][0] - centers_of_cities[i][0])**2 + (centers_of_cities[i+1][1] - centers_of_cities[i][1])**2)
	index = i+1
	for j in range(i+1, len(centers_of_cities)):
		dist = math.sqrt((centers_of_cities[j][0] - centers_of_cities[i][0])**2 + (centers_of_cities[j][1] - centers_of_cities[i][1])**2)
		if dist < min_distance:
			min_distance = dist
			index = j
	temp = centers_of_cities[i+1]
	centers_of_cities[i+1] = centers_of_cities[index]
	centers_of_cities[index] = temp


#CODE TO IDENTIFY WHITE COIN. CHANGE IF NEEDED.
circles = [list(map(int, lst)) for lst in circles[0, :]]
coords_pixelValues = {}
for circle in circles:
	coords_pixelValues[gray_cst[circle[0], circle[1]]] = circle
	sorted_pixelValues = sorted(coords_pixelValues.keys())
coins_list = []
for i in range(len(sorted_pixelValues) - 1):
	coins_list.append(coords_pixelValues[sorted_pixelValues[i]])


#CODE TO DETECT THE NODE NUMBER OF THE CAPITAL.
for center in centers_of_cities:
	if (center == capital_coordinates).all():
		capital_node = centers_of_cities.index(center) + 1


#CODE TO DETECT THE NODE NUMBER OF THE COINS.
center_coords = coords_pixelValues[sorted_pixelValues[-1]]
capital_coords = centers_of_cities[capital_node - 1]

node_number = []
for coin in coins_list:
	angle_differences = []
	for cities in centers_of_cities:
		if cities != capital_coords:
			dist1 = math.sqrt((cities[0] - center_coords[0])**2 + (cities[1] - center_coords[1])**2)
			dist2 = math.sqrt((coin[0] - cities[0])**2 + (coin[1] - cities[1])**2)
			dist3 = math.sqrt((coin[0] - center_coords[0])**2 + (coin[1] - center_coords[1])**2)
			city_angle = int((math.acos((dist1**2 + dist3**2 - dist2**2)/(2*dist1*dist3)))*(180/math.pi))
			angle_differences.append(city_angle)
		else:
			angle_differences.append(100)
	node_number.append(angle_differences.index(min(angle_differences)) + 1)


#SEQUENCE OF THE CITIES IN WHICH THE BOT HAS TO STOP.
node = capital_node + 1
sequence = []
while(True):
	if node%10 == capital_node:
		break
	elif node%10 in node_number:
		sequence.append(node%10)
	node += 1
	
#CODE THAT GETTS THE LIVE FEEDBACK ABOUT THE BOT AND SENDS THE NECESSARY SIGNALS TO THE BOT.
# m SIGNAL INDICATES MOVE.
# s SIGNAL INDICATES STOP AS THE CITY IS IN NEED OF AN AID OR SUPPLY.
# c SIGNAL INDICATES THE CAPITAL HAS ARRIVED AND ITS TIME TO END THE RUN.
PORT = "COM1"
BAUD_RATE = 9600
ser = serial.Serial(PORT, BAUD_RATE)
data = 'm'
ser.write(data.encode())

sequence += [capital_node]
j = 0
while(ret and j < len(sequence)):
	ret, img = cap.read()
	img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters = parameters)
	if corners != []:
		corner_points = corners[0][0]
		for i in range(len(corner_points)):
			corner_points[i, 0] = corner_points[i, 0]*(0.5)
			corner_points[i, 1] = corner_points[i, 1]*(0.5)
	img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
	capital_coordinates = corner_points.sum(axis=0)//4
	dist1 = math.sqrt((centers_of_cities[sequence[j] - 1][0] - capital_coordinates[0])**2 + (centers_of_cities[sequence[j] - 1][1] - capital_coordinates[1])**2)
	dist2 = math.sqrt((centers_of_cities[sequence[j] - 1][0] - center_coords[0])**2 + (centers_of_cities[sequence[j] - 1][1] - center_coords[1])**2)
	dist3 = math.sqrt((center_coords[0] - capital_coordinates[0])**2 + (center_coords[0] - capital_coordinates[1])**2)
	try:
		ang = int((math.acos((dist2**2 + dist3**2 - dist1**2)/(2*dist2*dist3)))*(180/math.pi))
	except:
		continue
	if ang < 5:
		if j != len(sequence) - 1:
			ser.write('s'.encode())
			current_city = sequence[j]
		else:
			ser.write('c'.encode())
		j += 1
