# -*- coding: utf-8 -*-
"""
Created on 5 May 2021

@author: Wong Wai Cheng

Description: This file contains the finetuned butterfly classifier.
"""

from functions import features_mobile1b, features_mobile1b_aug, plot_confusion_matrix, plot_result
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from joblib import dump
from joblib import load
from keras import models
from keras import layers
from keras import optimizers
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.layers import Dropout

train_path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/train'
test_path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/test-all'
# paths to offline augmented data
fliphori = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/fliphori'
flipverti = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/flipverti'
gamma = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/gamma'
rotate = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/rotate'
gaussian = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/gaussian'

train_size = 5600 # with offline augmentation 5600 * 6
test_size = 1552
batch_size = 10

# Extract features
from datetime import datetime
start = datetime.now()

train_features, train_labels = features_mobile1b(train_path, train_size, 224, 224, 7, 8) # without augmentation
test_features, test_labels = features_mobile1b(test_path, test_size, 224, 224, 7, 8)

# extract from manually augmented datasets
train_flip, train_labels_flip = features_mobile1b(fliphori, train_size, 224, 224, 7, 8)
train_flipv, train_labels_flipv = features_mobile1b(flipverti, train_size, 224, 224, 7, 8)
train_rotate, train_labels_rotate = features_mobile1b(rotate, train_size, 224, 224, 7, 8)
train_gamma, train_labels_gamma = features_mobile1b(gamma, train_size, 224, 224, 7, 8)
train_noise, train_labels_noise = features_mobile1b(gaussian, train_size, 224, 224, 7, 8)

# extract from online augmented dataset
# train_features_aug, train_labels_aug = features_mobile1b_aug(train_path, train_size, 224, 224, 7, 8)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Combine extracted features
train_features_combined = np.concatenate((train_features, train_flip, train_flipv, train_rotate, train_gamma, train_noise))
train_labels_combined = np.concatenate((train_labels, train_labels_flip, train_labels_flipv, train_labels_rotate, train_labels_gamma, train_labels_noise))

# combine with online augmented dataset
# train_features_combined = np.concatenate((train_features, train_features_aug))
# train_labels_combined = np.concatenate((train_labels, train_labels_aug))

"""
Softmax
"""
# lr_schedule = optimizers.schedules.ExponentialDecay(
#     initial_learning_rate=1e-4,
#     decay_steps=100
#     decay_rate=0.9)

starter_learning_rate = 0.01
end_learning_rate = 0.001
decay_steps = 10000
learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    starter_learning_rate,
    decay_steps,
    end_learning_rate,
    power=1)

epochs = 15
batch_size = 32
num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

acc_per_fold = []
loss_per_fold = []
precision_per_fold = []
recall_per_fold = []

# set up loop for 10 kfold cross validation
fold_no = 1
# build layers
from datetime import datetime
start = datetime.now()

for train, test in kfold.split(train_features_combined, train_labels_combined):
    mobile_soft_flat = models.Sequential()
    mobile_soft_flat.add(layers.Flatten(input_shape = (7,7,1024)))
    mobile_soft_flat.add(Dropout(0.7))
    mobile_soft_flat.add(layers.Dense(8, activation = 'softmax'))
    mobile_soft_flat.summary()
    # compile model
    mobile_soft_flat.compile(optimizer = optimizers.RMSprop(learning_rate=learning_rate_fn),
                          loss = 'categorical_crossentropy',
                          metrics = ['accuracy', tf.keras.metrics.Precision(),
                                     tf.keras.metrics.Recall()])

    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
    # run model training
    history_mobile_flat = mobile_soft_flat.fit(train_features_combined[train], train_labels_combined[train],
                                epochs = epochs,
                                batch_size = batch_size, steps_per_epoch=11200/32)
    # print scores for each fold
    scores = mobile_soft_flat.evaluate(train_features_combined[test], train_labels_combined[test], verbose = 0)
    print(f'Score for fold {fold_no}: {mobile_soft_flat.metrics_names[0]} of {scores[0]}; {mobile_soft_flat.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1])
    loss_per_fold.append(scores[0])
    precision_per_fold.append(scores[2])
    recall_per_fold.append(scores[3])

    plot_result(history_mobile_flat)

    fold_no = fold_no + 1

duration = datetime.now() - start
print("Training completed in time: ", duration)

# print score for each fold and average with s.d.
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]} - Precision: {precision_per_fold[i]} - Recall: {recall_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} \u00B1 {np.std(acc_per_fold)}')
print(f'> Loss: {np.mean(loss_per_fold)} \u00B1 {np.std(loss_per_fold)}')
print(f'> Precision: {np.mean(precision_per_fold)} \u00B1 {np.std(precision_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)} \u00B1 {np.std(recall_per_fold)}')
print('------------------------------------------------------------------------')

# save trained model for predicting test set later
mobile_soft_flat.save('final-model-softmax-manualaug.h5')
mobile_soft_flat = tf.keras.models.load_model('final-model-softmax-manualaug.h5')
# mobile_soft_flat.save('final-model-softmax-onlineaug-edited.h5')
# mobile_soft_flat = tf.keras.models.load_model('final-model-softmax-onlineaug-edited.h5')

# MobileNets-Flatten: Evaluate using test set
y_actual = np.argmax(test_labels, axis = 1)
labels = ['catopsilia-pomona', 'danaus-chrysippus', 'eurema-hecabe', 'graphium-doson',
            'graphium-sarpedon', 'junonia-hedonia', 'junonia-iphita', 'papilio-demoleus']

score = mobile_soft_flat.evaluate(test_features, test_labels)
print("%s: %.2f%%" % (mobile_soft_flat.metrics_names[1], score[1]*100))
y_pred = mobile_soft_flat.predict(test_features)
y_pred_argmax = np.argmax(y_pred, axis = 1)
confusion_matrix(y_actual, y_pred_argmax)
cm_mobile_flat = tf.math.confusion_matrix(labels = y_actual, predictions = y_pred_argmax).numpy()
cm_mobile_flat_df = pd.DataFrame(cm_mobile_flat,
                               index = labels,
                               columns = labels)

print(classification_report(y_actual, y_pred_argmax, digits = 4))

plot_confusion_matrix(cm=cm_mobile_flat, classes=labels, title='MobileNetV1 Finetune Softmax Offline Aug')
plt.tight_layout()
plt.savefig('final-model-softmax-manualaug.png')
plt.clf()

"""
SVM
"""
train_labels_svm = np.argmax(train_labels_combined, axis=1)
test_labels_svm = np.argmax(test_labels, axis=1)

# set up features and labels to be fed into SVM classifier
X_train, y_train = train_features_combined.reshape(33600, 7 * 7 * 1024), train_labels_svm
X_test, y_test = test_features.reshape(1552, 7 * 7 * 1024), test_labels_svm

num_folds = 10
kfold = KFold(n_splits=num_folds, shuffle=True)

acc_per_fold = []
precision_per_fold = []
recall_per_fold = []
f1_per_fold = []
y_pred_svm=[]
y_actual_svm=[]

# training
fold_no = 1
from datetime import datetime
start = datetime.now()

for train, test in kfold.split(X_train, y_train):

    mobile_svm = LinearSVC(dual=False, penalty='l2',loss='squared_hinge',C=0.1)
    mobile_svm.fit(X_train[train], y_train[train])
    y_pred = mobile_svm.predict(X_train[test])
    y_actual = y_train[test]
    metrics = classification_report(y_actual, y_pred, digits=4, output_dict=True)
    acc_per_fold.append(metrics['accuracy'])
    precision_per_fold.append(metrics['macro avg']['precision'])
    recall_per_fold.append(metrics['macro avg']['recall'])
    y_pred_svm.append(y_pred)
    y_actual_svm.append(y_train[test])
    fold_no = fold_no + 1

duration = datetime.now() - start
print("Training completed in time: ", duration)

# print score for each fold and average with s.d.
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
    print('------------------------------------------------------------------------')
    print(f'> Fold {i + 1} - Accuracy: {acc_per_fold[i]} - Precision: {precision_per_fold[i]} - Recall: {recall_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} \u00B1 {np.std(acc_per_fold)}')
print(f'> Precision: {np.mean(precision_per_fold)} \u00B1 {np.std(precision_per_fold)}')
print(f'> Recall: {np.mean(recall_per_fold)} \u00B1 {np.std(recall_per_fold)}')
print('------------------------------------------------------------------------')

dump(mobile_svm, 'final-model-svm-manualaug.joblib')
mobile_svm = load('final-model-svm-manualaug.joblib')
# dump(mobile_svm, 'final-model-svm-onlineaug-edited.joblib')
# mobile_svm = load('final-model-svm-onlineaug-edited.joblib')

# Evaluate using test set
result = mobile_svm.score(X_test, y_test)
print(result)

y_pred_svm_test = mobile_svm.predict(X_test)
matrix = confusion_matrix(y_test, y_pred_svm_test)
labels = ['catopsilia-pomona', 'danaus-chrysippus', 'eurema-hecabe', 'graphium-doson',
            'graphium-sarpedon', 'junonia-hedonia', 'junonia-iphita', 'papilio-demoleus']
cm_svm = tf.math.confusion_matrix(labels = y_test, predictions = y_pred_svm_test).numpy()
cm_svm_df = pd.DataFrame(cm_svm,
                     index = labels,
                     columns = labels)

print(classification_report(y_test, y_pred_svm_test, digits=4))

plot_confusion_matrix(cm=cm_svm, classes=labels, title='MobileNetV1 Finetune SVM Online Aug')
plt.tight_layout()
plt.savefig('final-model-svm-manualaug.png')
plt.clf()

###############################################################################################
                                        #End#
###############################################################################################