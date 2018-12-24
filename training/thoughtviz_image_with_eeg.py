import logging
import os
from random import randint
from keras import backend as K
import random
from PIL import Image
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
import pickle
import utils.data_input_util as inutil
from training.models.thoughtviz import *
from utils.image_utils import *
from utils.eval_utils import *


def train_gan(input_noise_dim, batch_size, epochs, data_dir, saved_classifier_model_file, model_save_dir, output_dir, classifier_model_file):

    K.set_learning_phase(False)
    # folders containing images used for training
    imagenet_folder = "./images/ImageNet-Filtered"
    num_classes = 10

    feature_encoding_dim = 100

    # load data and compile discriminator, generator models depending on the dataaset
    x_train, y_train, x_test, y_test = inutil.load_image_data(imagenet_folder, patch_size=(64, 64))
    print("Loaded Images Dataset.", )

    g_adam_lr = 0.00003
    g_adam_beta_1 = 0.5
   
    d_adam_lr = 0.00005
    d_adam_beta_1 = 0.5

    c = load_model(classifier_model_file)
    
    d = discriminator_model_rgb((64, 64), c)
    d_optim = Adam(lr=d_adam_lr, beta_1=d_adam_beta_1)
    d.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=d_optim)
    d.trainable = True

    g = generator_model_rgb(input_noise_dim, feature_encoding_dim)
    g_optim = Adam(lr=g_adam_lr, beta_1=g_adam_beta_1)
    g.compile(loss='categorical_crossentropy', optimizer=g_optim)

    d_on_g = generator_containing_discriminator(input_noise_dim, feature_encoding_dim, g, d)
    d_on_g.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=g_optim)

    g.summary()
    d.summary()
    
    eeg_data = pickle.load(open(os.path.join(data_dir, 'data.pkl'), "rb"))
    classifier = load_model(saved_classifier_model_file)
    classifier.summary()
    x_test = eeg_data[b'x_test']
    y_test = eeg_data[b'y_test']
    y_test = np.array([np.argmax(y) for y in y_test])
    layer_index = 9

    # keras way of getting the output from an intermediate layer
    get_nth_layer_output = K.function([classifier.layers[0].input], [classifier.layers[layer_index].output])

    layer_output = get_nth_layer_output([x_test])[0]

    for epoch in range(epochs):
        print("Epoch is ", epoch)

        print("Number of batches", int(x_train.shape[0]/batch_size))

        for index in range(int(x_train.shape[0]/batch_size)):
            # generate noise from a normal distribution
            noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))

            random_labels = np.random.randint(0, 10, batch_size)

            one_hot_vectors = [to_categorical(label, 10) for label in random_labels]
            
            eeg_feature_vectors = np.array([layer_output[random.choice(np.where(y_test == random_label)[0])] for random_label in random_labels])

            # get real images and corresponding labels
            real_images = x_train[index * batch_size:(index + 1) * batch_size]
            real_labels = y_train[index * batch_size:(index + 1) * batch_size]

            # generate fake images using the generator
            generated_images = g.predict([noise, eeg_feature_vectors], verbose=0)

            # discriminator loss of real images
            d_loss_real = d.train_on_batch(real_images, [np.array([1] * batch_size), np.array(real_labels)])
            # discriminator loss of fake images
            d_loss_fake = d.train_on_batch(generated_images, [np.array([0] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            d_loss = (d_loss_fake[0] + d_loss_real[0]) * 0.5

            d.trainable = False
            # generator loss
            g_loss = d_on_g.train_on_batch([noise, eeg_feature_vectors], [np.array([1] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            d.trainable = True

        # save generated images at intermediate stages of training
        if epoch % 100 == 0:
            image = combine_rgb_images(generated_images)
            image = image * 255.0
            img_save_path = os.path.join(output_dir, str(epoch) + "_g" + ".png")
            Image.fromarray(image.astype(np.uint8)).save(img_save_path)

        if epoch % 100 == 0:
            test_image_count = 50000
            test_noise = np.random.uniform(-1, 1, (test_image_count, input_noise_dim))
            test_labels = np.random.randint(0, 10, test_image_count)

            eeg_feature_vectors_test = np.array([layer_output[random.choice(np.where(y_test == test_label)[0])] for test_label in test_labels])
            test_images = g.predict([test_noise, eeg_feature_vectors_test], verbose=0)
            test_images = test_images * 255.0
            inception_score = get_inception_score([test_image for test_image in test_images], splits=10)

        print("Epoch %d d_loss : %f" % (epoch, d_loss))
        print("Epoch %d g_loss : %f" % (epoch, g_loss[0]))
        print("Epoch %d inception_score : %f" % (epoch, inception_score[0]))

        if epoch % 50 == 0:
            # save generator and discriminator models along with the weights
            g.save(os.path.join(model_save_dir, 'generator_' + str(epoch)), overwrite=True, include_optimizer=True)
            d.save(os.path.join(model_save_dir, 'discriminator_' + str(epoch)), overwrite=True, include_optimizer=True)


def train():
    dataset = 'Image'
    batch_size = 100
    run_id = 1
    epochs = 10000
    model_save_dir = os.path.join('./saved_models/thoughtviz_image_with_eeg/', dataset, 'run_' + str(run_id))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    output_dir = os.path.join('./outputs/thoughtviz_image_with_eeg/', dataset, 'run_' + str(run_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    classifier_model_file = os.path.join('./trained_classifier_models', 'classifier_' + dataset.lower() + '.h5')

    eeg_data_dir = os.path.join('../data/eeg/', dataset.lower())
    eeg_classifier_model_file = os.path.join('../models/eeg_models', dataset.lower(), 'run_final.h5')

    train_gan(input_noise_dim=100, batch_size=batch_size, epochs=epochs, splits_save_dir=eeg_data_dir, saved_classifier_model_file=eeg_classifier_model_file, model_save_dir=model_save_dir, output_dir=output_dir, classifier_model_file=classifier_model_file)


if __name__ == '__main__':
    train()
