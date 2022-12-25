"""
To avoid anu possible bias due to other aspetc of US image,
such as the name of the machine and the color map, the images in
'evaluation former are slightly modify.
In particular two rectangular are superimpose on the left side and on the 
top right of the image
"""
import os
import argparse
import numpy as np
import cv2 

def sample_n(path):
	"""
	Return the sample number in the selected folder

	Parameters
	----------
	path : string
		folder's path

	Returns
	-------
	sample_num : list
		list of number
	"""
	sample_num = []
	for files in os.listdir(path):
		ext = files.split('.')[-1]
		if ext == 'png' : sample_num.append(int(files.split('.')[0].split('_')[-1]))
	return sample_num

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Futher modification on evaluate image folder')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	args = parser.parse_args()

	# path 
	evaluation_path = 'GAN' + '/' + args.attribute + '/' + args.clas + '/evaluation'
	image_path = evaluation_path + '/sample_evaluation_34.png'

	sample_num = sample_n(evaluation_path)
	
	# Reading an image in grayscale mode
	for n in sample_num:
		image = cv2.imread(evaluation_path + f'/sample_evaluation_{n}.png', 0)

		# Window name in which image is displayed
		window_name = f'Image_{n}'
		
		start_point_1 = (143, 58)  # top left corner of rectangle
		end_point_1 = (178, 426)    # the bottom right corner of rectangle
		start_point_2 = (458, 58)  # top left corner of rectangle
		end_point_2 = (512, 132)    # the bottom right corner of rectangle
		
		color = (0, 0, 0)        # Black color in BGR
		thickness = -1           # thickness of -1 will fill the entire shape
		
		# Using cv2.rectangle() method
		# Draw a rectangle of black color of thickness -1 px
		image = cv2.rectangle(image, start_point_1, end_point_1, color, thickness)
		image = cv2.rectangle(image, start_point_2, end_point_2, color, thickness)

		cv2.imshow(window_name, image) 

		cv2.waitKey(1000)
		cv2.destroyAllWindows()

