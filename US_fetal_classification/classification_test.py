"""
Evaluation of trained model on the test set
"""
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
import seaborn as sns
import matplotlib.pyplot as plt     
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

from makedir import *
from classification_utils import true_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main for classification problem of US fetal image')
    parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
    parser.add_argument("model_name", type=str, help="""Model name: 'MobileNetV2'; 'VGG_16'; 'DenseNet_169""")

    args = parser.parse_args()

    ## LOAD TEST SET
    test_path = 'Images_classification_' + args.attribute +'/test/'
    test_dataset = image_dataset_from_directory(test_path,
                                                shuffle=False,
                                                batch_size = 1,
                                                image_size=(224,224),
                                                interpolation='bilinear')

    models_path = 'Images_classification_' + args.attribute + '/' + 'models'
    model = tf.keras.models.load_model(models_path + '/'+ args.model_name)
    # print(model.summary())
    # print(test_dataset.class_names)
    
    # ## TRUE AND PREDICTED LABELS
    # true_labels = true_labels(test_path, test_dataset)
    # prediction = model.predict(test_dataset, verbose=1)
    # predicted_labels = np.argmax(prediction, axis=-1)

    # ## CONFUSION MATRIX
    # confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)
    # cf = confusion_matrix.numpy()
    
    # normalization_dict = {}
    # for label, clas in enumerate(test_dataset.class_names):
    #     normalization = len(os.listdir(test_path + '/' + clas))
    #     normalization_dict[label] = len(os.listdir(test_path + '/' + clas))
    
    # cf_norm = []
    # for i in range(len(os.listdir(test_path))):
    #     cf_norm.append(cf[i]/ normalization_dict[i])
    
    # cf = np.array(cf_norm)
    # print(cf)

    # ## K-ACCURACY
    # m_1 =  tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    # m_1.update_state(true_labels, prediction)
    # m_3 =  tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    # m_3.update_state(true_labels, prediction)

    # print(f'ACCURACY = {cf.diagonal().mean():.3f} +- {cf.diagonal().std(ddof=1):.3f}')
    # print(f'top-1 error = {1-m_1.result().numpy()}')
    # print(f'top-3 error = {1-m_3.result().numpy()}')

    ## SAVE STATISTIC
    save_path = 'Images_classification_' + args.attribute +'/models/' + args.model_name + '_'
    print(save_path)
    if args.model_name + '_' in os.listdir('Images_classification_' + args.attribute +'/models'):
    	pass
    else: smart_makedir(save_path)
    np.save(save_path + '/confusion', cf)
    np.save(save_path + '/prediction', np.array(prediction))

    fig, ax= plt.subplots(nrows=1, ncols=1, figsize=(10,10), num='confusion_matrix')

    sns.heatmap(cf, annot=True, fmt='.3g', ax=ax, cmap='Blues') #annot=True to annotate cells, ftm='g' to disable scientific notation
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix');

    classes = test_dataset.class_names
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes)
    
    plt.savefig(save_path + 'confusion_matrix')
    plt.show()
    with open(save_path + '/' +'statistic.txt', 'w', encoding='utf-8') as file:
        file.write(f'ACCURACY = {cf.diagonal().mean():.3f} +- {cf.diagonal().std(ddof=1):.3f}')
        file.write(f'top-1 error = {1-m_1.result().numpy()}')
        file.write(f'top-3 error = {1-m_3.result().numpy()}')
        
