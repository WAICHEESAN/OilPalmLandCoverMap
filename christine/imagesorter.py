# -*- coding: utf-8 -*-
"""
Created on 2 May 2021

@author: Wong Wai Cheng

Description: This file contains the code for sorting images into different species.
"""

import os, shutil
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from joblib import load
from functions import mobilebase1b

batch_size = 10
img_width_mobile, img_height_mobile = 224, 224

"""
Image Sorter with Softmax
"""
def visualize_predictions(classifier):
    path ='/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/tosort/'
    output_path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/tosort/'
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        # print(f)
        img = image.load_img(f, target_size=(224,224))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        # Extract features
        features = mobilebase1b.predict(img_tensor.reshape(1, img_width_mobile, img_height_mobile, 3))

        # Make prediction
        try:
            prediction = classifier.predict(features)
        except:
            prediction = classifier.predict(features.reshape(1, 7 * 7 * 1024))

        # Show picture
        plt.imshow(img_tensor)
        plt.show()

        # Write prediction
        if np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 0:
            print(prediction, 'catopsilia-pomona')
            if not os.path.exists(output_path + 'catopsilia-pomona'):
                    os.makedirs(output_path + 'catopsilia-pomona')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='catopsilia pomona',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 1:
            print(prediction, 'danaus-chrysippus')
            if not os.path.exists(output_path + 'danaus-chrysippus'):
                    os.makedirs(output_path + 'danaus-chrysippus')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='danaus-chrysippus',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 2:
            print(prediction, 'eurema-hecabe')
            if not os.path.exists(output_path + 'eurema-hecabe'):
                    os.makedirs(output_path + 'eurema-hecabe')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='eurema-hecabe',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 3:
            print(prediction, 'graphium-doson')
            if not os.path.exists(output_path + 'graphium-doson'):
                    os.makedirs(output_path + 'graphium-doson')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='graphium-doson',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 4:
            print(prediction, 'graphium-sarpedon')
            if not os.path.exists(output_path + 'graphium-sarpedon'):
                    os.makedirs(output_path + 'graphium-sarpedon')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='graphium-sarpedon',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 5:
            print(prediction, 'junonia-hedonia')
            if not os.path.exists(output_path + 'junonia-hedonia'):
                    os.makedirs(output_path + 'junonia-hedonia')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='junonia-hedonia',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 6:
            print(prediction, 'junonia-iphita')
            if not os.path.exists(output_path + 'junonia-iphita'):
                    os.makedirs(output_path + 'junonia-iphita')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='junonia-iphita',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) > 9.8e-1 and np.argmax(prediction) == 7:
            print(prediction, 'papilio-demoleus')
            if not os.path.exists(output_path + 'papilio-demoleus'):
                    os.makedirs(output_path + 'papilio-demoleus')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='papilio-demoleus',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif np.argmax(prediction) < 9.8e-1:
            print(prediction, 'unidentifiable')
            if not os.path.exists(output_path + 'unidentifiable'):
                    os.makedirs(output_path + 'unidentifiable')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='unidentifiable',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)

"""
Image Sorter with SVM
"""
def visualize_predictions_svm(classifier):
    path ='/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/tosort/'
    output_path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/tosort/'
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        # print(f)
        img = image.load_img(f, target_size=(224,224))
        img_tensor = image.img_to_array(img)
        img_tensor /= 255.

        # Extract features
        features = mobilebase1b.predict(img_tensor.reshape(1, 224, 224, 3))

        # Make prediction
        try:
            prediction = classifier.predict(features)
        except:
            prediction = classifier.predict(features.reshape(1, 7 * 7 * 1024))

        # Show picture
        plt.imshow(img_tensor)
        plt.show()

        # Write prediction
        if prediction == 0:
            print(prediction, 'catopsilia-pomona')
            if not os.path.exists(output_path + 'catopsilia-pomona'):
                    os.makedirs(output_path + 'catopsilia-pomona')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='catopsilia-pomona',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 1:
            print(prediction, 'danaus-chrysippus')
            if not os.path.exists(output_path + 'danaus-chrysippus'):
                    os.makedirs(output_path + 'danaus-chrysippus')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='danaus-chrysippus',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 2:
            print(prediction, 'eurema-hecabe')
            if not os.path.exists(output_path + 'eurema-hecabe'):
                    os.makedirs(output_path + 'eurema-hecabe')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='eurema-hecabe',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 3:
            print(prediction, 'graphium-doson')
            if not os.path.exists(output_path + 'graphium-doson'):
                    os.makedirs(output_path + 'graphium-doson')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='graphium-doson',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 4:
            print(prediction, 'graphium-sarpedon')
            if not os.path.exists(output_path + 'graphium-sarpedon'):
                    os.makedirs(output_path + 'graphium-sarpedon')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='graphium-sarpedon',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 5:
            print(prediction, 'junonia-hedonia')
            if not os.path.exists(output_path + 'junonia-hedonia'):
                    os.makedirs(output_path + 'junonia-hedonia')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='junonia-hedonia',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 6:
            print(prediction, 'junonia-iphita')
            if not os.path.exists(output_path + 'junonia-iphita'):
                    os.makedirs(output_path + 'junonia-iphita')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='junonia-iphita',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        elif prediction == 7:
            print(prediction, 'papilio-demoleus')
            if not os.path.exists(output_path + 'papilio-demoleus'):
                    os.makedirs(output_path + 'papilio-demoleus')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='papilio-demoleus',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)
        else:
            print(prediction, 'unidentifiable')
            if not os.path.exists(output_path + 'unidentifiable'):
                    os.makedirs(output_path + 'unidentifiable')
            full_output_path = "{op}{lbl}/{fn}".format(op=output_path, lbl='unidentifiable',
                                                           fn=file_names_array[files.index(f)])
            shutil.copyfile(f, full_output_path)

# softmax
model = tf.keras.models.load_model('/Users/chris/Desktop/MasterofDataScience/thesis_related/projectbutterfly/final-model-softmax-manualaug.h5')
visualize_predictions(model)

# svm
svm_model = load('/Users/chris/Desktop/MasterofDataScience/thesis_related/projectbutterfly/final-model-svm-manualaug.joblib')
visualize_predictions_svm(svm_model)

###############################################################################################
                                        #End#
###############################################################################################