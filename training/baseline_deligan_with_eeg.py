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
from training.models.deligan import *
from utils.image_utils import *


def train_gan(dataset, input_noise_dim, batch_size, epochs, data_dir, saved_classifier_model_file, model_save_dir, output_dir):

    K.set_learning_phase(False)
    # folders containing images used for training
    char_fonts_folders = ["./images/Char-Font"]
    num_classes = 10

    feature_encoding_dim = 100

    # load data and compile discriminator, generator models depending on the dataaset
    if dataset == 0:
        x_train, y_train, x_test, y_test = inutil.load_digit_data()
        print("Loaded Digits Dataset.", )

    if dataset == 1:
        x_train, y_train, x_test, y_test = inutil.load_char_data(char_fonts_folders, resize_shape=(28, 28))
        print("Loaded Characters Dataset.", )

    adam_lr = 0.00005
    adam_beta_1 = 0.5

    d = discriminator_model((28, 28), num_classes)
    d_optim = Adam(lr=adam_lr, beta_1=adam_beta_1)
    d.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=d_optim)
    d.trainable = True

    g = generator_model(input_noise_dim, feature_encoding_dim)
    g_optim = Adam(lr=adam_lr, beta_1=adam_beta_1)
    g.compile(loss='categorical_crossentropy', optimizer=g_optim)

    d_on_g = generator_containing_discriminator(input_noise_dim, feature_encoding_dim, g, d)
    d_on_g.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=g_optim)

    g.summary()
    d.summary()
    
    splits = pickle.load(open(os.path.join(data_dir, 'data.pkl'), "rb"))
    classifier = load_model(saved_classifier_model_file)
    classifier.summary()
    x_test = splits[b'x_test']
    y_test = splits[b'y_test']
    y_test = [np.argmax(y) for y in y_test]
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

            #conditioned_noise = []
            #for i in range(batch_size):
            #    conditioned_noise.append(np.append(noise[i], eeg_feature_vectors[i]))
            #conditioned_noise = np.array(conditioned_noise)

            # get real images and corresponding labels
            real_images = x_train[index * batch_size:(index + 1) * batch_size]
            real_labels = y_train[index * batch_size:(index + 1) * batch_size]

            # generate fake images using the generator
            #generated_images = g.predict(conditioned_noise, verbose=0)
            generated_images = g.predict([noise, eeg_feature_vectors], verbose=0)	   
  
            # discriminator loss of real images
            d_loss_real = d.train_on_batch(real_images, [np.array([1] * batch_size), np.array(real_labels)])
            # discriminator loss of fake images
            d_loss_fake = d.train_on_batch(generated_images, [np.array([0] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            d_loss = (d_loss_fake[0] + d_loss_real[0]) * 0.5

            # save generated images at intermediate stages of training
            if index % 250 == 0:
                image = combine_images(generated_images)
                image = image * 127.5 + 127.5
                img_save_path = os.path.join(output_dir, str(epoch) + "_g_" + str(index) + ".png")
                Image.fromarray(image.astype(np.uint8)).save(img_save_path)

            d.trainable = False
            # generator loss
            #g_loss = d_on_g.train_on_batch(conditioned_noise, [np.array([1] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            g_loss = d_on_g.train_on_batch([noise, eeg_feature_vectors], [np.array([1] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            d.trainable = True

        print("Epoch %d d_loss : %f" % (epoch, d_loss))
        print("Epoch %d g_loss : %f" % (epoch, g_loss[0]))

        # save generator and discriminator models along with the weights
        g.save(os.path.join(model_save_dir, 'generator_' + str(epoch)), overwrite=True, include_optimizer=True)
        d.save(os.path.join(model_save_dir, 'discriminator_' + str(epoch)), overwrite=True, include_optimizer=True)


def train():
    folder_name_mapping = {0: 'Digit', 1: 'Char'}
    dataset = 0
    batch_size = 100
    run_id = 2
    epochs = 100
    model_save_dir = os.path.join('./saved_models/baseline_deligan_with_eeg/', folder_name_mapping[dataset], 'run_' + str(run_id))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    output_dir = os.path.join('./outputs/baseline_deligan_with_eeg/', folder_name_mapping[dataset], 'run_' + str(run_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    eeg_data_dir = os.path.join('../data/eeg/', folder_name_mapping[dataset].lower())
    eeg_classifier_model_file = os.path.join('../models/eeg_models', folder_name_mapping[dataset].lower(), 'run_final.h5')

    train_gan(dataset=dataset, input_noise_dim=100, batch_size=batch_size, epochs=epochs, data_dir=eeg_data_dir, saved_classifier_model_file=eeg_classifier_model_file, model_save_dir=model_save_dir, output_dir=output_dir)


if __name__ == '__main__':
    train()
