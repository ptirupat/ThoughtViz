import os
from random import randint

from PIL import Image
from keras.optimizers import Adam

import utils.data_input_util as inutil
from training.models.ac_gan import *
from utils.eval_utils import *
from utils.image_utils import *


def train_gan(input_noise_dim, batch_size, epochs, model_save_dir, output_dir):

    # folders containing images used for training
    imagenet_folder = "./images/ImageNet-Filtered"
    
    num_classes = 10

    # load data and compile discriminator, generator models depending on the dataaset
    x_train, y_train, x_test, y_test = inutil.load_image_data(imagenet_folder, patch_size=(64, 64))
    print("Loaded Images Dataset.", )

    adam_lr = 0.001
    adam_beta_1 = 0.5

    d = discriminator_model_rgb((64, 64), num_classes)
    d_optim = Adam(lr=adam_lr, beta_1=adam_beta_1)
    d.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=d_optim)
    d.trainable = True

    g = generator_model_rgb(input_noise_dim + num_classes)
    g_optim = Adam(lr=adam_lr, beta_1=adam_beta_1)
    g.compile(loss='categorical_crossentropy', optimizer=g_optim)

    d_on_g = generator_containing_discriminator(input_noise_dim + num_classes, g, d)
    d_on_g.compile(loss=['binary_crossentropy','categorical_crossentropy'], optimizer=g_optim)

    g.summary()
    d.summary()
    
    for epoch in range(epochs):
        print("Epoch is ", epoch)

        print("Number of batches", int(x_train.shape[0]/batch_size))

        for index in range(int(x_train.shape[0]/batch_size)):
            # generate noise from a normal distribution
            noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))

            random_labels = [randint(0, 9) for i in range(batch_size)]

            one_hot_vectors = [to_categorical(label, 10) for label in random_labels]

            conditioned_noise = []
            for i in range(batch_size):
                conditioned_noise.append(np.append(noise[i], one_hot_vectors[i]))
            conditioned_noise = np.array(conditioned_noise)

            # get real images and corresponding labels
            real_images = x_train[index * batch_size:(index + 1) * batch_size]
            real_labels = y_train[index * batch_size:(index + 1) * batch_size]

            # generate fake images using the generator
            generated_images = g.predict(conditioned_noise, verbose=0)

            # discriminator loss of real images
            d_loss_real = d.train_on_batch(real_images, [np.array([1] * batch_size), np.array(real_labels)])
            # discriminator loss of fake images
            d_loss_fake = d.train_on_batch(generated_images, [np.array([0] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            d_loss = (d_loss_fake[0] + d_loss_real[0]) * 0.5

            # save generated images at intermediate stages of training
            if index % 250 == 0:
                image = combine_rgb_images(generated_images)
                image = image * 255.0
                img_save_path = os.path.join(output_dir, str(epoch) + "_g_" + str(index) + ".png")
                Image.fromarray(image.astype(np.uint8)).save(img_save_path)

            d.trainable = False
            # generator loss
            g_loss = d_on_g.train_on_batch(conditioned_noise, [np.array([1] * batch_size), np.array(one_hot_vectors).reshape(batch_size, num_classes)])
            d.trainable = True

        test_image_count = 50000
        test_noise = np.random.uniform(-1, 1, (test_image_count, input_noise_dim))
        test_labels = [randint(0, 9) for i in range(test_image_count)]
        one_hot_vectors_test = [to_categorical(label, 10) for label in test_labels]

        conditioned_noise_test = []
        for i in range(test_image_count):
            conditioned_noise_test.append(np.append(test_noise[i], one_hot_vectors_test[i]))
        conditioned_noise_test = np.array(conditioned_noise_test)

        test_images = g.predict(conditioned_noise_test, verbose=0)
        test_images = test_images * 255.0
        
        if epoch % 10 == 0:        
            inception_score = get_inception_score([test_image for test_image in test_images], splits=10)

        print("Epoch %d d_loss : %f" % (epoch, d_loss))
        print("Epoch %d g_loss : %f" % (epoch, g_loss[0]))
        print("Epoch %d inception_score : %f" % (epoch, inception_score[0]))

        # save generator and discriminator models along with the weights
        g.save(os.path.join(model_save_dir, 'generator_' + str(epoch)), overwrite=True, include_optimizer=True)
        d.save(os.path.join(model_save_dir, 'discriminator_' + str(epoch)), overwrite=True, include_optimizer=True)


def train():
    dataset = 'Image'
    batch_size = 32
    run_id = 5
    epochs = 1000
    model_save_dir = os.path.join('./saved_models/baseline_acgan_image/', dataset, 'run_' + str(run_id))
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    output_dir = os.path.join('./outputs/baseline_acgan_image/', dataset, 'run_' + str(run_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_gan(input_noise_dim=100, batch_size=batch_size, epochs=epochs, model_save_dir=model_save_dir, output_dir=output_dir)


if __name__ == '__main__':
    train()
