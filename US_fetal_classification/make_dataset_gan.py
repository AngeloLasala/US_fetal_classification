"""
CAM for GANs. Exploit the CAM algorithms for Classification task to built
the sutable dataset fro Pix2Pix cGAN.
input-->CAM mask      target-->US_images 
"""
import os
import sys
import argparse
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image_dataset_from_directory


from image_utils import smart_plot, get_img_array
from classification_utils import *

def true_labels_from_dataset(dataset):
	"""
	Return the true labels of dataset_from_directory.
	The dataset MUST BE sorted in alphanumeric order (shaffle=False) and
	batch_size MUST BE equals to 1

	Parameters
	----------
	dataset : tenserflow dataset
		input dataset

	Returns
	-------
	true_labels : np.array
		array of true labels
	"""
	true_labels = []
	for image_bach, label_batch in dataset.as_numpy_iterator():
		image_bach = image_bach[0,:,:,:]
		label_batch = label_batch[0]
		true_labels.append(label_batch)
	true_labels = np.array(true_labels)	
	return true_labels

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make dtataset for pix2pix GAN exploit CAM')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	parser.add_argument("train", type=str, help="""name of inner folder, train_number""")

	args = parser.parse_args()

	## IMPORT IMAGE
	metadata_path = 'FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv'
	data_frame = pd.read_csv(metadata_path, index_col=None)

	## CREATE FOLDER TO SAVE IMAGES
	gan_path = 'GAN/' + args.attribute + '/' + args.clas 
	smart_makedir(gan_path + '/train')
	smart_makedir(gan_path + '/test')


	## MODEL ######################################################################################
	## LOAD TRAINED MODEL
	model_par = model_parameter('VGG_16') # we need only the preprocession function
	models_path = "Images_classification_" + args.attribute + "/models/" + 'VGG_16_/' + args.train
	model = keras.models.load_model(models_path + '/'+ 'VGG_16')

	## WEIGHT OF CLASSIFICATION LAYER and MODEL FOR THE CAM
	class_weights = model.layers[-1].get_weights()[0]
	
	# select the nested layer in the functional one insede  the 'vgg16' layer
	last_conv_layer_name = 'block5_conv3'
	x = model.get_layer('vgg16').get_layer(last_conv_layer_name).output
	a = model.get_layer('vgg16').get_layer('input_1').input
	grad_model = tf.keras.Model(inputs = [model.inputs, a], outputs=[x,  model.output])
	
	## SELECT ONLY TRUE PREDICTION IMAGE
	print("SELECT ONLY TRUE PREDICTION IMAGE")
	test_path = 'Images_classification_' + args.attribute +'/test/'
	test_dataset = image_dataset_from_directory(test_path,
												shuffle=False,
												batch_size = 1,
												image_size=(224,224),
												interpolation='bilinear',
												seed = 42)
	image_batch, label_batch = test_dataset.as_numpy_iterator().next()
	
	true_labels = true_labels_from_dataset(test_dataset)
		
	prediction = np.load(models_path + '/'+ 'prediction.npy')
	predicted_labels = np.argmax(prediction, axis=-1)	
	
	## TRUE POSISIVE OF SELECTED CLASS
	numb_class = test_dataset.class_names.index(args.clas)
	pos_true_labels = np.where(true_labels == numb_class)
	predicted_on_pos = predicted_labels[pos_true_labels]
	true_positive = np.where(predicted_on_pos == numb_class)[0]
	print(true_positive)
	
	#########################################################################################################
	main_path = "Images_classification_" + args.attribute + "/test/"+ args.clas +"/"
	patient_list = os.listdir(main_path)
	patient_list.sort()
	print(f'total images: {len(patient_list)}')
	for index in true_positive:
		print(index)
		patient = patient_list[index].split('.')[0]
		image_path = main_path + patient + '.png'
		print(image_path)
	
		#  predicted class for our input image with respect to the activations of the last conv layer
		preprocess_input = model_par['preprocessing']
		img_array = preprocess_input(get_img_array(image_path, size=model_par['IMG_SIZE']))
		
		with tf.GradientTape() as tape:
			last_conv_layer_output, preds = grad_model([img_array, img_array])

		## CAM - SPLIT INPUT(CAM) AND REAL IMAGE(IMG)
		heatmap = cam_heatmap(preds, class_weights, last_conv_layer_output)
		input_image, real_image = save_cam(image_path, heatmap, save_it=False)	
		concatenate_image = np.stack((input_image, real_image), axis=0)
		concatenate_image = np.reshape(concatenate_image,
										(2*real_image.shape[0], real_image.shape[1], real_image.shape[2]))

		## save the image in train and test
		if np.random.uniform(0,1) <= 0.05:
			save_array_as_image(concatenate_image, gan_path + '/test/' + f'sample_{index}')
		else:
			save_array_as_image(concatenate_image, gan_path + '/train/' + f'sample_{index}')

	