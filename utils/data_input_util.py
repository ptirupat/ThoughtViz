
import os
from random import randint

import PIL
import numpy as np
from PIL import Image
from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.feature_extraction import image


# mapping class labels to a number
CHARACTER_CLASSES = {'A': 0, 'C': 1, 'F': 2, 'H': 3, 'J': 4, 'M': 5, 'P': 6, 'S': 7, 'T': 8, 'Y': 9}

IMAGE_CLASSES = {'Apple': 0, 'Car': 1, 'Dog': 2, 'Gold': 3, 'Mobile': 4, 'Rose': 5, "Scooter": 6, 'Tiger': 7, 'Wallet': 8, 'Watch': 9}


def randomize(samples, labels):
    if type(samples) is np.ndarray:
        permutation = np.random.permutation(samples.shape[0])
        shuffle_samples = samples[permutation]
        shuffle_lables = labels[permutation]
    else:
        permutation = np.random.permutation(len(samples))
        shuffle_samples = [samples[i] for i in permutation]
        shuffle_lables = [labels[i] for i in permutation]

    return (shuffle_samples, shuffle_lables)


def load_digit_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def load_char_data(char_fonts_folders, resize_shape):

    images = []
    labels = []
    for char_fonts_folder in char_fonts_folders:
        for char_folder in os.listdir(char_fonts_folder):
            char_class = CHARACTER_CLASSES[char_folder]
            for char_img in os.listdir(os.path.join(char_fonts_folder, char_folder)):
              file_path = os.path.join(char_fonts_folder, char_folder, char_img)
              img = Image.open(file_path).resize(resize_shape, PIL.Image.NEAREST).convert('L')
              img_array = 255 - np.array(img)
              images.append(img_array)
              labels.append(char_class)

    images, labels = randomize(images, labels)
    train_size = int(3 * len(images)/4)
    x_train, y_train = np.array(images[0: train_size]), np.array(labels[0: train_size])
    x_test, y_test = np.array(images[train_size:]), np.array(labels[train_size:])

    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_test = (x_test.astype(np.float32) - 127.5) / 127.5

    x_train = x_train[:, :, :, None]
    x_test = x_test[:, :, :, None]
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return x_train, y_train, x_test, y_test


def load_image_data(imagenet_folder, patch_size):

    images = []
    labels = []
    
    for image_folder in os.listdir(imagenet_folder):
        image_class = IMAGE_CLASSES[image_folder]
        for image_file in os.listdir(os.path.join(imagenet_folder, image_folder)):
            file_path = os.path.join(imagenet_folder, image_folder, image_file)
            img = Image.open(file_path).convert('RGB').resize(patch_size, PIL.Image.ANTIALIAS)
            img_array = np.array(img)
            img_array = img_array/255.0
            images.append(img_array)
            labels.append(image_class)
            images.append(np.flip(img_array, 1))
            labels.append(image_class)
    
    print(len(images), len(labels))
    images, labels = randomize(images, labels)

    images = np.array(images)
    labels = np.array(labels)
    train_size = int(3 * len(images)/4)
    x_train, y_train = images[0: train_size], labels[0: train_size]
    x_test, y_test = images[train_size:], labels[train_size:]

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x_train, y_train, x_test, y_test


