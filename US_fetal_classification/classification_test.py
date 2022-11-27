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
from classification_utils import true_labels, statistical_analysis

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main for classification problem of US fetal image')
    parser.add_argument("attribute", type=str, help="Attributo to cliification task: 'Plane' or 'Brain_plane'")
    parser.add_argument("model_name", type=str, help="""Model name: 'MobileNetV2'; 'VGG_16'; 'DenseNet_169""")
    parser.add_argument("train", type=str, help="""name of inner folder, train_number""")

    args = parser.parse_args()

    ## LOAD TEST SET
    test_path = 'Images_classification_' + args.attribute +'/test/'
    test_dataset = image_dataset_from_directory(test_path,
                                                shuffle=False,
                                                batch_size = 1,
                                                image_size=(224,224),
                                                interpolation='bilinear')

    models_path = 'Images_classification_' + args.attribute +'/models/' + args.model_name + '_'
    models_path = models_path + '/' + args.train
    model = tf.keras.models.load_model(models_path + '/'+ args.model_name)
    print(model.summary())
    print(test_dataset.class_names)

    ## TEST vs VALIDATION curves
    accuracy = np.load(models_path + '/history_accuracy.npy')
    val_accuracy = np.load(models_path + '/history_val_accuracy.npy')
    loss = np.load(models_path + '/history_loss.npy')
    val_loss = np.load(models_path + '/history_val_loss.npy')

    print("VALIDATION FOR MODEL SELECTION")
    print(f"accuracy on the last epoch: {val_accuracy[-1]}")
    print(f"losses on the last epoch: {val_loss[-1]}")

    
    ## TRUE AND PREDICTED LABELS
    true_labels = true_labels(test_path, test_dataset)
    prediction = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(prediction, axis=-1)

    ## CONFUSION MATRIX
    confusion_matrix = tf.math.confusion_matrix(true_labels, predicted_labels)
    cf = confusion_matrix.numpy()
    
    normalization_dict = {}
    for label, clas in enumerate(test_dataset.class_names):
        normalization = len(os.listdir(test_path + '/' + clas))
        normalization_dict[label] = len(os.listdir(test_path + '/' + clas))
    
    cf_norm = []
    for i in range(len(os.listdir(test_path))):
        cf_norm.append(cf[i]/ normalization_dict[i])
    
    cf = np.array(cf_norm)
    print(cf)

    ## K-ACCURACY
    m_1 =  tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)
    m_1.update_state(true_labels, prediction)
    m_3 =  tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3)
    m_3.update_state(true_labels, prediction)

    ## PRECISION, RECALL and F1-SCORE
    if args.attribute == 'Plane':
        report_dict = statistical_analysis(true_labels, predicted_labels, test_dataset)

    print(f'ACCURACY = {cf.diagonal().mean():.3f} +- {cf.diagonal().std(ddof=1):.3f}')
    print(f'top-1 error = {1-m_1.result().numpy()}')
    print(f'top-3 error = {1-m_3.result().numpy()}')

    ## SAVE STATISTIC
    np.save(models_path + '/confusion', cf)
    np.save(models_path + '/prediction', np.array(prediction))

    fig1, ax1= plt.subplots(nrows=1, ncols=1, figsize=(10,10), num='confusion_matrix')

    sns.heatmap(cf, annot=True, fmt='.3g', ax=ax1, cmap='Blues') #annot=True to annotate cells, ftm='g' to disable scientific notation
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title('Confusion Matrix');

    classes = test_dataset.class_names
    ax1.xaxis.set_ticklabels(classes)
    ax1.yaxis.set_ticklabels(classes)
    plt.savefig(models_path + '/confusion_matrix')
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8,8), num='Train_Val_curves')
    acc = [0.] + accuracy
    val_acc = [0.] + val_accuracy

    ax[0].plot(acc, label='Training Accuracy')
    ax[0].plot(val_acc, label=f'Validation Accuracy: last epoch {val_accuracy[-1]:.4f}')
    ax[0].legend(loc='lower right')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_ylim([min(plt.ylim()),1])
    ax[0].set_title('Training and Validation Accuracy')

    ax[1].plot(loss, label='Training Loss')
    ax[1].plot(val_loss, label=f'Validation Loss: last epoch {val_loss[-1]:.4f}')
    ax[1].legend(loc='upper right')
    ax[1].set_ylabel('Loss')
    ax[1].set_ylim([0,1.0])
    ax[1].set_title('Training and Validation Loss')
    ax[1].set_xlabel('epoch')
    plt.savefig(models_path + '/' + 'Train_Val_curves')
    
    with open(models_path + '/' +'statistic.txt', 'w', encoding='utf-8') as file:
        file.write(f'ACCURACY = {cf.diagonal().mean():.3f} +- {cf.diagonal().std(ddof=1):.3f} \n')
        file.write(f'top-1 error = {1-m_1.result().numpy()} \n')
        file.write(f'top-3 error = {1-m_3.result().numpy()} \n')
        