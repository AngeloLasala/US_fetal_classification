"""
Make datasetset folder for classification task.

UPDATE: Brain_plane need more accurate splitting procedure, see paper.
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from makedir import *
from image_utils import *
from classification_utils import dataset_folder
import Hyperparameter as hyper

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Make folder and subfold for attribute classification task')
	parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
	parser.add_argument("-make_folder", action='store_true', help="Actual make folder, default=False")
	args = parser.parse_args()

	## path of images and metadata folder
	images_path = 'FETAL_PLANES_ZENODO/Images'
	metadata_path = 'FETAL_PLANES_ZENODO/FETAL_PLANES_DB_data.csv'

	## Data frame 
	data_frame = pd.read_csv(metadata_path, index_col=None)

	## Make Folder
	if args.make_folder == True:
		dataset_folder(data_frame, args.attribute)

	## Check consisten of subfolder split
	folder_path_train = 'Images_classification'+'_'+args.attribute+'/train'
	folder_path_test = 'Images_classification'+'_'+args.attribute+'/test'
	classes = os.listdir(folder_path_train)

	print('TRAIN')
	tot = 0
	for clas in classes:
		path = folder_path_train + '/' + clas
		tot +=len(os.listdir(path))
		print(f'{clas}: {len(os.listdir(path))}')
	print(f'Total: {tot}\n')

	print('TEST')
	tot = 0 
	for clas in classes:
		path = folder_path_test + '/' + clas
		tot +=len(os.listdir(path))
		print(f'{clas}: {len(os.listdir(path))}')
	print(f'Total: {tot}\n')
	