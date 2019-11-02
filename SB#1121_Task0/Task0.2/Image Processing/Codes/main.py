import cv2
import numpy as np
import os

def partA():
	images_path = os.path.join(os.getcwd(), '..', 'Images')
	f_ref = open(os.path.join(os.getcwd(), '..', 'Generated', 'stats.csv'), 'w')
	for image in os.listdir(images_path):
		if '.jpg' in image:
			image_path = os.path.join(images_path, image)
			img = cv2.imread(image_path, 1)
			rows, columns, channels = img.shape
			b_v = img[rows//2, columns//2, 0]
			g_v = img[rows//2, columns//2, 1]
			r_v = img[rows//2, columns//2, 2]
			f_ref.write(','.join(list(map(str, [image, rows, columns, channels, b_v, g_v, r_v]))) + '\n')


def partB():
	images_path = os.path.join(os.getcwd(), '..', 'Images')
	image_path = os.path.join(images_path, 'cat.jpg')
	img = cv2.imread(image_path, 1)
	img[:, :, 0] = 0
	img[:, :, 1] = 0
	cv2.imwrite(os.path.join(os.getcwd(), '..', 'Generated', 'cat_red.jpg'), img)


def partC():
	images_path = os.path.join(os.getcwd(), '..', 'Images')
	image_path = os.path.join(images_path, 'flowers.jpg')
	img = cv2.imread(image_path, 1)
	img_new = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
	img_new[:, :, 3] = 0.5*255
	cv2.imwrite(os.path.join(os.getcwd(), '..', 'Generated', 'flowers_alpha.png'), img_new)

def partD():
	images_path = os.path.join(os.getcwd(), '..', 'Images')
	image_path = os.path.join(images_path, 'horse.jpg')
	img = cv2.imread(image_path, 1)
	B, G, R = img[:, :, 0], img[:, :, 1], img[:, :, 2]
	result = 0.3*R + 0.59*G + 0.11*B
	cv2.imwrite(os.path.join(os.getcwd(), '..', 'Generated', 'horse_gray.jpg'), result)


partA()
partB()
partC()
partD()