"""
Class Activation Map of trained model.
"""
import os
import argparse
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from image_utils import smart_plot, get_img_array, normalization
from classification_utils import *


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Class Activation Maps for Plane and Brain_plane classification')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	parser.add_argument("train", type=str, help="""name of inner folder, train_number""")

	args = parser.parse_args()

	## IMPORT IMAGE
	metadata_path = 'FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv'
	data_frame = pd.read_csv(metadata_path, index_col=None)

	main_path = "Images_classification_" + args.attribute + "/test/"+ args.clas +"/"
	patient_list = os.listdir(main_path)
	index = np.random.randint(0, int(len(patient_list)))
	patient = patient_list[index].split('.')[0]
	image_path = main_path + patient + '.png'

	## LOAD TRAINED MODEL
	model_par = model_parameter('VGG_16') # we need only the preprocession function
	models_path = "Images_classification_" + args.attribute + "/models/" + 'VGG_16_/' + args.train
	model = keras.models.load_model(models_path + '/'+ 'VGG_16')
	print(model.summary(expand_nested=True))

	## WEIGHT OF CLASSIFICATION LAYER
	class_weights = model.layers[-1].get_weights()[0]
	
	## LAST CONVOLUTIONAL LAYER AND PREDICTIO 
	# select the nested layer in the functional one insede  the 'vgg16' layer
	last_conv_layer_name = 'block5_conv3'
	x = model.get_layer('vgg16').get_layer(last_conv_layer_name).output
	a = model.get_layer('vgg16').get_layer('input_1').input

	grad_model = tf.keras.Model(inputs = [model.inputs, a], outputs=[x,  model.output])
	# print(grad_model.summary())

	# Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
	preprocess_input = model_par['preprocessing']
	img_array = preprocess_input(get_img_array(image_path, size=model_par['IMG_SIZE']))
	
	with tf.GradientTape() as tape:
		last_conv_layer_output, preds = grad_model([img_array, img_array])
		print(last_conv_layer_output.shape)
		print(preds)

	## CAM 
	predicted_class = tf.argmax(preds[0])
	class_weights = class_weights[:,predicted_class]
	class_weights = np.expand_dims(class_weights, axis=-1)

	last_conv_layer_output = last_conv_layer_output[0,:,:,:]

	heatmap = last_conv_layer_output @ class_weights
	heatmap = np.squeeze(heatmap)
	heatmap = normalization(heatmap, 0, 1)
	print(heatmap.shape)

	## SAVE AND REPORT THE RESULTS
	if args.attribute == 'Plane': train_class = ('Fetal abdomen', 'Fetal brain', 'Fetal femur', 'Fetal thorax', 'Maternal cervix', 'Other')
	if args.attribute == 'Brain_plane': train_class = ('Not a Brain', 'Trans-cerebellum', 'Trans-thalamic', 'Trans-ventricular')
	print("PREDICTION")
	print(f"model prediction: {preds[0]}")
	print(f"predicted class: {predicted_class} - {train_class[predicted_class]}")
	smart_plot(data_frame, image_path)

	cam_patient_path = "Images_classification_" + args.attribute + "/CAM_1/" + patient
	smart_makedir(cam_patient_path)
	save_cam(image_path, heatmap, cam_path = cam_patient_path + "/cam.jpg")
	with open(cam_patient_path + '/' +'prediction.txt', 'w', encoding='utf-8') as file:
		file.write("PREDICTION\n")
		file.write(f"True class: {args.clas}\n\n")
		file.write(f"model prediction: {preds[0]}\n")
		file.write(f"predicted class: {predicted_class} - {train_class[predicted_class]}")
