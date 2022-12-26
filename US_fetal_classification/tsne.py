"""
Visualization of multidimentional data throught t-distributed Stochastic Neighbor Embedding
(t-SNE).
"""
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing import image_dataset_from_directory

from makedir import *
from image_utils import normalization
from classification_utils import *
from gan_utils import *
import matplotlib.pyplot as plt



def model_brain_plane(model_path):
	"""
	Load trained model for the t-SNE
	"""
	model = tf.keras.models.load_model(model_path)
	# print(model.summary(expand_nested=True))

	## WEIGHT OF CLASSIFICATION LAYER
	class_weights = model.layers[-1].get_weights()[0]
	
	## LAST CONVOLUTIONAL LAYER AND PREDICTIO 
	# select the nested layer in the functional one insede  the 'vgg16' layer
	last_conv_layer_name = 'global_average_pooling2d'
	x = model.get_layer(last_conv_layer_name).output
	a = model.get_layer('input_2').input

	grad_model = tf.keras.Model(inputs = [a], outputs=[x])
	# print(grad_model.summary())
	return grad_model

def custom_model(shape):
	"""
	Create a model for the t-SNE using VGG_16 pretrained on imagenet
	"""
	input_shape = shape + (3,)
    

	base_model = tf.keras.applications.vgg16.VGG16(input_shape=input_shape,
													include_top=False, # <== Important!!!!
													weights='imagenet') # From imageNet


	# create the input layer (Same as the imageNetv2 input size)
	inputs = tf.keras.Input(shape=input_shape) 

	# data preprocessing using the same weights the model was trained on
	# Already Done -> preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
	preprocess_input = tf.keras.applications.vgg16.preprocess_input
	x = preprocess_input(inputs) 

	# set training to False to avoid keeping track of statistics in the batch norm layer
	x = base_model(x, training=False) 

	# Add the new Binary classification layers
	# use global avg pooling to summarize the info in each channel
	x = tfl.GlobalAveragePooling2D()(x)
	model = tf.keras.Model(inputs, x)

	return model

	
if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='t-SNE evaluation of Cam-cGan model')
	parser.add_argument("attribute", type=str, help="Attribute to classification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("clas", type=str, help="Class to classification task: example 'Fetal brain' or 'Trans-cerebellum'")
	parser.add_argument('-extrapolation', action='store_true', help='extrapolate the feature. default=False')
	args = parser.parse_args()

	## Main path of Classification folder
	gan_path = 'GAN/' + args.attribute + '/' + args.clas 
	models_path = "Images_classification_" + args.attribute + "/models/" + 'VGG_16_/' + 'train_11'

	tsne_path = gan_path + '/tsne'
	if 'tsne' in os.listdir(gan_path):
			pass
	else:
		smart_makedir(tsne_path)
	
	

	# feature extrapolation
	INPUT_SHAPE = (224,224)
	input_shape = INPUT_SHAPE + (3,)
	
	if args.extrapolation:
		base_model = custom_model(INPUT_SHAPE)
		brain_model = model_brain_plane(models_path + '/'+ 'VGG_16')
		print(brain_model.summary())

		features, test_dataset = feature_extraction(brain_model, tsne_path + '/train')
		print(features.shape)
		np.save(tsne_path + '/features_GAP_brain', features)

	
	## LOAD FEATURE
	features = np.load(tsne_path + '/features_GAP_brain.npy')
	print(features.shape)

	## PCA
	n_components = 50
	pca = PCA(n_components=n_components)
	print(f"Fraction of variance preserved: {pca.fit(features).explained_variance_ratio_.sum():.2f}")
	features_pca = pca.fit_transform(features)
	print(features_pca.shape)

	## t-SNE
	tsne = TSNE(n_components=2, learning_rate='auto', init='pca', perplexity=50, n_iter=6000, verbose=1).fit_transform(features_pca)
	tx = normalization(tsne[:, 0],0,1)
	ty = normalization(tsne[:, 1],0,1)

	## Plots the t-SNE
	test_dataset = image_dataset_from_directory(tsne_path + '/train',
                                                shuffle=False,
                                                batch_size = 1,
                                                image_size = INPUT_SHAPE,
                                                interpolation='bilinear')
	classes = test_dataset.class_names
	colors = ['C0', 'C1', 'C2', 'C3', 'C4']
	labels = [] 
	for img, label in test_dataset.take(2056):
		labels.append(label.numpy())

	plt.figure()
	for idx, c in enumerate(colors):
		indices = [i for i, l in enumerate(labels) if idx == l]
		current_tx = np.take(tx, indices)
		current_ty = np.take(ty, indices)
		print(len(indices))
		plt.scatter(current_tx, current_ty, c=c, label=classes[idx])

	plt.legend()
	plt.show()
