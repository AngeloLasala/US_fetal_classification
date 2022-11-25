"""
Explore US images and metadata file 
"""
import os
import numpy as np 
import pandas as pd 
from PIL import Image
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import imageio
from makedir import *
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
	image_row = data.loc[data['Image_name'] == image_name]
	print(image_row)
	print(f'pixel = {img.shape}')

	# features
	patient_num = image_row['Patient_num'].unique()[0] 
	plane = image_row['Plane'].unique()[0] 
	brain_plane = image_row['Brain_plane'].unique()[0]

	# create the numpy array and plot it
	fig, ax = plt.subplots(nrows=1, ncols=1, num = f'Patient {patient_num}: {plane} {brain_plane}__{image_name}')
	ax.set_title(f'{plane} - {brain_plane}')
	ax.imshow(img)

	return img

def brain_plot(data, n_examples = 5):
	"""
	Plot 5 different items of different brain_plane

	Parameters
	----------
	data : dataframe

	n_example : integer,	
		numerb of example for each brain_plane
	"""

	fig, ax = plt.subplots(nrows=n_examples, ncols=5, figsize=(10,10), num='Simle Images Visualization')

	for i, value in enumerate(data['Brain_plane'].unique()):
		print(i, value)
		images = image_paths(data, 'Brain_plane', value)
		index = np.random.randint(0, len(images), size=(n_examples))
		
		ax[0,i].set_title(value)
		for j, ii in enumerate(index):
			img = imageio.imread(images[ii])
			ax[j,i].imshow(img)

def split_data(data):
	"""
	Split the dataset in train and test looking at the "Train " attribute
	ATTECTION: with original data image this fuction create two variable (train_set and test_set)
	which could fill RAM. Before appling split_data, I suggest to preprocess the images.

	Parameters
	----------
	data : datatframe

	Returns
	-------
	train_set : list of array
		list of training images
	
	test_set : list of array
		list of testing images
	"""
	train_paths = image_paths(data, 'Train ', 1)
	test_paths = image_paths(data, 'Train ', 0)

	train_set = []
	for path in train_paths:
		img = imageio.imread(path)
		train_set.append(img)

	test_set = []
	for path in test_paths:
		img = imageio.imread(path)
		test_set.append(img)
	
	return train_set, test_set

def check_anomalies(data, dataset, data_string):
	"""
	Check the images with differents distribuction:
	- 3D shape instead of 2D shape
	- table instead of US image

	Parameters
	----------
	data : data_frame

	dataset: list
		set train or test set, it comes from 'split_data' fuction

	data_string : string ('train' or 'test')
		string identification of dataset list 

	Returns
	-------
	indeces : list
		list of position of strange image in train or test dataset
	"""
	if data_string == 'train': i_data = 1
	if data_string == 'test': i_data = 0


	indeces = []
	for i, img in enumerate(dataset):
		if len(img.shape)>2: 
			print(i, img.shape)
			indeces.append(i)
	print(f'Train: {len(indeces)} 3D-shape imgas')

	for ii in indeces:
		smart_plot(data, image_paths(data, 'Train ', 0)[ii])

	return indeces

def normalization(image, a, b):
	"""
	Normalization function for image in the interval (a,b).

	x' = (b-a) (x -x_max)/(x_max-x_min) + a

	MinMaxScaling  a=image.min(), b=image.max()

	Parameters
	----------
	image: PIL.image or numpy array
		image's array

	a : float
		bottom bound of the interval

	b : float
		upper bound of the interval

	Return
	------
	new_image: np.array
		resize of the image
	"""
	image = np.array(image, dtype=np.float32)

	x_max = image.max()
	x_min = image.min()
	
	new_image = (b-a) * (image - x_min)/(x_max-x_min) + a
	

	return new_image

def processing_image(image,
					input_size = (224,224),
					resampling = Image.Resampling.LANCZOS,
					scaler = 'Normalization',
					a = 0,
					b = 1,
					):
	"""
	Process original image and return the input shape for learning. 
	I can do this with Tensorflow too.

	The main steps are:
	1) Reshape: set the total number of pixel in the form (height, width)
	2) Normalization: normalize the pixel value
	3) Convet: from greyscale to RGB mode

	Parameters
	----------
	image : image from Image.open,
		imported image
	
	input_size : tuple 
		input size to DL architecture

	scaler: string,
		'MinMax', 'StandardScaler', 'Normalization'

	Result
	------
	image_to_save : image
		processed image

	"""
	# 1) Reshape
	img_reshape = image.resize(input_size, resampling)

	# 2) convert to RGB
	img_rgb = img_reshape.convert('RGB')

	# 2) Normalization
	if scaler == 'MinMaxScaler':
		scaler = MinMaxScaler()
		scaler.fit(img_rgb)
		img_scaled = scaler.transform(img_rgb)

	if scaler == 'StandardScaler':
		scaler = StandardScaler()
		scaler.fit(img_rgb)
		img_scaled = scaler.transform(img_rgb) 

	if scaler == 'Normalization':
		img_scaled = normalization(img_rgb, a, b)

	image_to_save = img_scaled

	return image_to_save 

def preprocess_data(data_frame,
				    images_path = 'FETAL_PLANES_ZENODO/Images', 
					folder_name = 'Preprocess_image',
					scaler='Normalization',  
					a=-1, 
					b=1):
	"""
	Make directory with preprocessed data.
	"""
	folder_name = 'Preprocess_image'
	smart_makedir(folder_name)

	for index in range(data_frame.shape[0]):
		print(index)
		image_path = images_path +'/'+ data_frame.iloc[index]['Image_name']+'.png'
		img = Image.open(image_path)
		img_to_save = processing_image(img, scaler=scaler, a=a, b=b)
		np.save(folder_name + '/'+ data_frame.iloc[index]['Image_name'], img_to_save)

def target_label(data_frame, attribute):
	"""
	Return the data set of target label of given attribute.
	Target label: [0,...,1,...0] with shape (lenght_attributes_classes)

	Parameters
	----------
	data_frame: pandas data frame
		data frame

	attribute: string
		attibute to classify
	
	Return
	------
	y_data : numpy.array
		data set of target label
	"""
	N = len(data_frame[attribute].unique())
	plane_dict = {plane_cls:i for i, plane_cls  in enumerate(data_frame[attribute].unique())}
	print(plane_dict)
	y_data = []
	for index in range(data_frame.shape[0]):
		y_image = np.zeros((N))
		image_plane_cls = data_frame.iloc[index][attribute]
		y_image[plane_dict[image_plane_cls]] = 1
		y_data.append(y_image)

	return np.array(y_data)

def get_img_array(img_path, size):
	"""
	Return the image in numpy array ready to be 'predicted'.  trasform the images in
	size (1, model_par['IMG_SIZE'][0], model_par['IMG_SIZE'][0], 3)
	used in cam.py 

	Parameters
	----------
	img_path : string
		path of images

	size : tuple
		output size of image 

	Return
	------
	array : numpy array 
		4-d array
	"""
    # `img` is a PIL image of size model_par['IMG_SIZE']
	img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (model_par['IMG_SIZE'][0], model_par['IMG_SIZE'][0], 3)
	array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, model_par['IMG_SIZE'][0], model_par['IMG_SIZE'][0], 3)
	array = np.expand_dims(array, axis=0)
	return array


if __name__ == '__main__':
	images_path = 'FETAL_PLANES_ZENODO/Images'
	metadata_path = 'FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv'

	## Basic Information
	data_frame = pd.read_csv(metadata_path, index_col=None)
	image_list = os.listdir(images_path)

	print('ATTRIBUTES')
	print(f'{data_frame.columns}')
	print('==============================================')

	plane_dict = data_number_example(data_frame, 'Plane')
	print()
	
	# List of image's path of selected attibutes and value
	index = 50
	fetal_brain_image = image_paths(data_frame, 'Plane', 'Fetal brain')

	# Simple Image rapreentation
	brain_plot(data_frame, 6)

	## Check incongruent samples: 3D array and strage image shape
	print('Start split train and test: takes about 2.0 minutes')
	train_set, test_set = split_data(data_frame)
	print('OK! splitting complete \n')

	# Incongruent image on test  
	stange_img_pos = check_anomalies(data_frame, train_set, 'train')
	print(f'anomalies: {len(stange_img_pos)}')

	## Check normalization
	img = Image.open(images_path+'/Patient01792_Plane3_1_of_1.png')
	aa = np.random.rand(224,224)*255

	img_r= normalization(img, a=-5, b=1)

	# ## PLOTS ################################
	plt.imshow(img)
	plt.show()
