"""
Explore US images and metadata file 
"""
import os
import numpy as np 
import pandas as pd 
from PIL import Image
import imageio
import matplotlib.pyplot as plt

def data_number_example(data, attribute):
	"""
	Total number of value of selected attribute

	Parameters
	----------
	df : dataframe

	attribute : (string) 
		dataframe's attribute (feature)

	Return
	------
	val_dict: dict
		total numer of example of selected attribute
	"""
	print(attribute)

	val_dict = {}
	for value in data[attribute].unique():
		df = data[data[attribute]==value]
		print(f'{value}: {df.shape[0]}')
		val_dict[value] = df.shape[0]
	
	return val_dict

def image_paths(data, attribute, value):
	"""
	Return the image_path of selected value about selected attribute

	Parameters
	----------
	data : dataframe

	attribute : string,
		selected attribute

	value : string or intger
		selected value of attribute. Have to be in attribute domain

	Return
	------

	list_image : list
		list of image's paths
	"""

	list_image = list()
	if (value in data[attribute].unique()): 
		
		data[data[attribute] == value]
		patient = data[data[attribute] == value]['Image_name']
		for i in patient:
			path_image = 'FETAL_PLANES_ZENODO/Images/'+ i +'.png'
			list_image.append(path_image)

	else : 
		raise ValueError(f'the value {value} is not a {attribute} value')
	 
	return list_image

def smart_plot(data,image_path):
	"""
	Plot image and print important feature of image

	Parameters
	----------
	data : dataframe

	image_path : string,
		path of selected image
	"""

	img = imageio.imread(image_path)

	image_name = image_path.split('/')[-1].split('.')[0]
	image_row = data.loc[data_frame['Image_name'] == image_name]
	print(image_row)
	print(f'pixel = {img.shape}')

	# features
	patient_num = image_row['Patient_num'].unique()[0] 
	plane = image_row['Plane'].unique()[0] 
	brain_plane = image_row['Brain_plane'].unique()[0]

	# create the numpy array and plot it
	fig, ax = plt.subplots(nrows=1, ncols=1, num = f'Patient {patient_num}: {plane} {brain_plane}')
	ax.set_title(f'{plane} - {brain_plane}')
	ax.imshow(img)
	plt.show()

	return img


	

if __name__ == '__main__':
	images_path = 'FETAL_PLANES_ZENODO/Images'
	metadata_path = 'FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv'

	data_frame = pd.read_csv(metadata_path, index_col=None)
	image_list = os.listdir(images_path)

	print('ATTRIBUTES')
	print(f'{data_frame.columns}')
	print('==============================================')

	plane_dict = data_number_example(data_frame, 'Plane')
	print()
	
	index = 50
	fetal_brain_image = image_paths(data_frame, 'Plane', 'Fetal brain')
	

	## PLOTS
	imageio.imread(fetal_brain_image[index])
	smart_plot(data_frame, fetal_brain_image[index])

	
	




