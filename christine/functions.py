# -*- coding: utf-8 -*-
"""
Created on 13 March 2021

@author: Wong Wai Cheng

Description: This file contains all the general functions for running the butterfly classifier.
Aim: Build a classifier for classifying species of butterflies via
1. Transfer learning - feature extractor
2. Transfer learning with finetuning
"""

import os, glob, shutil, random, itertools
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet_v2 import MobileNetV2
import efficientnet.keras as efn
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
import matplotlib.pyplot as plt
import numpy as np

"""
For sorting datasets into train, validation and test folders
"""
def clear_used_samples(source_list, sample_list):
    for i in sample_list:
        source_list.remove(i)

def prep_data(directory, species, train_size, test_size):
    os.chdir(directory)
    if os.path.isdir(f'train/{species}') is True:
        shutil.rmtree(f'train/{species}')
    os.makedirs(f'train/{species}')
    # if os.path.isdir(f'valid/{species}') is True:
    #     shutil.rmtree(f'valid/{species}')
    # os.makedirs(f'valid/{species}')
    if os.path.isdir(f'test/{species}') is True:
        shutil.rmtree(f'test/{species}')
    os.makedirs(f'test/{species}')
    total_list = glob.glob(f'{species}/*')
    training_list = random.sample(total_list, train_size)
    clear_used_samples(total_list, training_list)
    [shutil.copy2(pic, f'train/{species}') for pic in training_list]
    # validation_list = random.sample(total_list, validate_size)
    # clear_used_samples(total_list, validation_list)
    # [shutil.copy2(pic, f'valid/{species}') for pic in validation_list]
    test_list = random.sample(total_list, test_size)
    [shutil.copy2(pic, f'test/{species}') for pic in test_list]

###############################################################################################
                                        #Light models#
###############################################################################################
"""
MobileNetV1 - 7,7,1024
"""
# each model has their own image input size
img_width_mobile, img_height_mobile = 224, 224
img_width_eff, img_height_eff = 244, 244

# set up base model, remove output, freeze layers
mobilebase1 = MobileNet(input_shape = (img_width_mobile, img_height_mobile, 3),
                         include_top = False,
                         weights = 'imagenet')
mobilebase1.trainable = False
batch_size = 10

# MobileNetV1 without augmentation
def features_mobile1(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                   )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1024))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = mobilebase1.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

# MobileNetV1 with augmentation
def features_aug_mobile1(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen_aug = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip = True,
                                 vertical_flip = True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise
                                 )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1024))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen_aug.flow_from_directory(directory,
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = mobilebase1.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels

"""
MobileNetV2 - 7,7,1280
"""
mobilebase2 = MobileNetV2(input_shape = (img_width_mobile, img_height_mobile, 3),
                         include_top = False,
                         weights = 'imagenet')
mobilebase2.trainable = False
batch_size = 10

# MobileNetV2 without augmentation
def features_mobile2(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                   )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1280))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = mobilebase2.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# MobileNetV2 with augmentation
def features_aug_mobile2(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen_aug = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip = True,
                                 vertical_flip = True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise)
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1280))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen_aug.flow_from_directory(directory,
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = mobilebase2.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels

"""
EfficientNetB0 - 8,8,1280
"""
effbase0 = efn.EfficientNetB0(input_shape = (img_width_eff, img_height_eff, 3),
                         include_top = False,
                         weights = 'imagenet')
effbase0.trainable = False
batch_size = 10

# EfficientNetB0 without augmentation
def features_eff0(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                   )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1280))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = effbase0.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# EfficientNetB0 with augmentation
def features_aug_eff0(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen_aug = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip = True,
                                 vertical_flip = True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise)
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1280))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen_aug.flow_from_directory(directory,
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size,
                                            class_mode = 'categorical')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = effbase0.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
        return features, labels

###############################################################################################
                                        # Deep models #
###############################################################################################
"""
VGG16 - 7,7,512
"""
vgg16base = VGG16(input_shape=(224,224,3),
                  include_top=False,
                  weights='imagenet')

vgg16base.trainable=False
batch_size=10

# without augmentation
def features_vgg16(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                 )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 512))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg16base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# with augmentation
def features_vgg16_aug(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip=True,
                                 vertical_flip= True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise)
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 512))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg16base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

"""
VGG19 - 7,7,512
"""
vgg19base = VGG19(input_shape=(224,224,3),
                  include_top=False,
                  weights='imagenet')

vgg19base.trainable=False
batch_size=10

# without augmentation
def features_vgg19(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                   )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 512))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg19base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# with augmentation
def features_vgg19_aug(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise)
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 512))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = vgg19base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

"""
ResNet50 - 7,7,2048
"""
resnetbase = ResNet50(input_shape=(224,224,3),
                  include_top=False,
                  weights='imagenet')

resnetbase.trainable=False
batch_size=10

# without augmentation
def features_resnet(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                   )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 2048))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = resnetbase.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# with augmentation
def features_resnet_aug(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise
                                 )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 2048))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = resnetbase.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels


###############################################################################################
                                        #Finetune#
###############################################################################################
"""
Finetuned MobileNetV1
"""
mobilebase1b = MobileNet(input_shape = (img_width_mobile, img_height_mobile, 3),
                        include_top = False,
                        weights = 'imagenet')
mobilebase1b.trainable = True
for layer in mobilebase1b.layers[:-20]: # set number of trainable layers
    layer.trainable = False

# without augmentation
def features_mobile1b(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                 )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1024))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = mobilebase1b.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

# with augmentation
def add_noise(img):
    '''Add random noise to an image'''
    VARIABILITY = 50
    deviation = VARIABILITY*random.random()
    noise = np.random.normal(0, deviation, img.shape)
    img += noise
    np.clip(img, 0., 255.)
    return img

def features_mobile1b_aug(directory, sample_count, img_width, img_height, outputdim, num_classes):
    datagen = ImageDataGenerator(rescale = 1./255,
                                 rotation_range=45,
                                 horizontal_flip=True,
                                 vertical_flip=True,
                                 brightness_range=[0.2,1.0],
                                 preprocessing_function=add_noise
                                 )
    features = np.zeros(shape = (sample_count, outputdim, outputdim, 1024))
    labels = np.zeros(shape = (sample_count, num_classes)) # match number of classes

    generator = datagen.flow_from_directory(directory,
                                              target_size = (img_width, img_height),
                                              batch_size = batch_size,
                                              class_mode = 'categorical')

    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = mobilebase1b.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features, labels

###############################################################################################
                                        #Functions for Plots#
###############################################################################################
"""
For plotting accuracy and loss curves
"""
def plot_result(history):
    acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    # val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, acc, label = 'Training accuracy')
    # plt.plot(epochs, val_acc, label = 'Validation accuracy')
    plt.title('Training accuracy')
    plt.legend(loc='lower right')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, loss, label = 'Training loss')
    # plt.plot(epochs, val_loss, label = 'Validation loss')
    plt.title('Training loss')
    plt.legend(loc='upper right')

    plt.show()

"""
For plotting confusion matrix
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion Matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

###############################################################################################
                                        #End#
###############################################################################################