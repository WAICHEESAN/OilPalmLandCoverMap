# -*- coding: utf-8 -*-
"""
Created on 10 May 2021

@author: Wong Wai Cheng

Description: This file contains the functions for basic image transformations for data augmentation purposes.
Aim: Perform manual basic image manipulations on training dataset for Malaysia butterflies.
"""

import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import os, glob, shutil
import matplotlib
%matplotlib inline

"""
Plot the transformations for 1 sample image
"""
image = imageio.imread('/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/train/0/0009catopsilia-pomona.jpg')
ia.imshow(image)

# Rotate
rotate = iaa.Affine(rotate=(-45))
rotated_image = rotate.augment_image(image)
ia.imshow(rotated_image)
imageio.imwrite('/Users/chris/Desktop/rotate0009catopsilia-pomona.jpg', rotated_image)

# Gaussian noise
gaussian_noise = iaa.AdditiveGaussianNoise(10,20)
noise_image = gaussian_noise.augment_image(image)
ia.imshow(noise_image)
imageio.imwrite('/Users/chris/Desktop/noise0009catopsilia-pomona.jpg', noise_image)

# Flip horizontal
flip_hr = iaa.Fliplr(p=1.0)
flip_hr_image = flip_hr.augment_image(image)
ia.imshow(flip_hr_image)
imageio.imwrite('/Users/chris/Desktop/fliphr0009catopsilia-pomona.jpg', flip_hr_image)

# Flip vertical
flip_vr=iaa.Flipud(p=1.0)
flip_vr_image= flip_vr.augment_image(image)
ia.imshow(flip_vr_image)
imageio.imwrite('/Users/chris/Desktop/flipvr0009catopsilia-pomona.jpg', flip_vr_image)

# Gamma contrast
image = imageio.imread('/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/train/0/0009catopsilia-pomona.jpg')
contrast=iaa.GammaContrast(gamma=1.5)
contrast_image =contrast.augment_image(image)
ia.imshow(contrast_image)
imageio.imwrite('/Users/chris/Desktop/contrast0009catopsilia-pomona.jpg', contrast_image)

"""
Transform training datasets:
1. Flip horizontal and vertical
2. Adjust contrast of image
3. Add Gaussian noise
4. Rotate image
"""

# Convert folder of class 4 and 7 to RGB first.
from PIL import Image
def convert_rgb(path):
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)

    for f in files:
        image = Image.open(f)
        rgb_im = image.convert('RGB')
        rgb_im.save(f)

convert_rgb('/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/0') # folder for class 0 - 7

# Rotate
def rotation(path):
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        image = imageio.imread(f)
        rotate = iaa.Affine(rotate=(-45))
        rotated_image = rotate.augment_image(image)
        imageio.imwrite(os.path.join(path, f), rotated_image)

path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/rotate/4' # folder for class 0 - 7
rotation(path=path)

# Horizontal flip
def fliphr(path):
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        image = imageio.imread(f)
        flip_hr = iaa.Fliplr(p=1.0)
        flip_hr_image = flip_hr.augment_image(image)
        imageio.imwrite(os.path.join(path, f), flip_hr_image)

path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/fliphori/4' # folder for class 0 - 7
fliphr(path=path)

# Gamma contrast
def gamma(path):
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        image = imageio.imread(f)
        contrast=iaa.GammaContrast(gamma=1.5)
        contrast_image =contrast.augment_image(image)
        imageio.imwrite(os.path.join(path, f), contrast_image)

path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/gamma/4' # folder for class 0 - 7
gamma(path=path)

# Gaussian noise
def gaussian(path):
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        image = imageio.imread(f)
        gaussian_noise = iaa.AdditiveGaussianNoise(5,5)
        noise_image = gaussian_noise.augment_image(image)
        imageio.imwrite(os.path.join(path, f), noise_image)

path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/gaussian/7' # folder for class 0 - 7
gaussian(path=path)

# Vertical flip
def flipverti(path):
    files = []
    file_names_array = []
    for r, d, file_names in os.walk(path):
        for file in file_names:
            if file.lower().endswith(('.png','.jpg','.jpeg')):
                files.append(os.path.join(r, file))
                file_names_array.append(file)
    for f in files:
        image = imageio.imread(f)
        flip_vr=iaa.Flipud(p=1.0)
        flip_vr_image= flip_vr.augment_image(image)
        imageio.imwrite(os.path.join(path, f), flip_vr_image)

path = '/Users/chris/Desktop/MasterofDataScience/thesis_related/dataset/final/augmented/train/flipverti/7' # folder for class 0 - 7
flipverti(path=path)

###############################################################################################
                                        #End#
###############################################################################################