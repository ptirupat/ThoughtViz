import keras
from keras.layers import Dense, Flatten, MaxPooling2D
from keras.layers import Reshape, LeakyReLU, Dropout
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.convolutional import Conv2DTranspose
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model


def generator_model_rgb(noise_dim):
    noise_input = keras.layers.Input(shape=(noise_dim,))
    x = BatchNormalization(momentum=0.8)(noise_input)
    x = Dense(512 * 4 * 4, activation="relu")(x)
    x = Reshape((4, 4, 512))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=256, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=128, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=64, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2DTranspose(filters=3, kernel_size=5, strides=2, padding='same', activation='relu')(x)
    output = Activation("tanh")(x)

    return Model(inputs=[noise_input], outputs=[output])


def discriminator_model_rgb(input_img_shape, output_dim):
    img_input = keras.layers.Input(shape=(input_img_shape[0], input_img_shape[1], 3))
    x = Conv2D(16, (3, 3), strides=2)(img_input)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), strides=2)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(512, (3, 3), strides=1)(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)

    fake = Dense(1, activation='sigmoid')(x)
    aux = Dense(output_dim, activation='softmax')(x)
    return Model(inputs=[img_input], outputs=[fake, aux])


def generator_model(noise_dim):
    noise_input = keras.layers.Input(shape=(noise_dim,))
    x = Dense(1024, activation="tanh")(noise_input)
    x = BatchNormalization(momentum=0.8)(x)
    x = Dense(128 * 7 * 7, activation="tanh")(x)
    x = Reshape((7, 7, 128))(x)
    x = UpSampling2D()(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Conv2D(64, kernel_size=5, padding="same", activation="tanh")(x)
    x = UpSampling2D()(x)
    x = Conv2D(1, kernel_size=3, padding="same")(x)
    output = Activation("tanh")(x)

    return Model(inputs=[noise_input], outputs=[output])


def discriminator_model(input_img_shape, output_dim):
    img_input = keras.layers.Input(shape=(input_img_shape[0], input_img_shape[1], 1))
    x = Conv2D(64, (5, 5), padding="same")(img_input)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(128, (5, 5))(x)
    x = Activation('tanh')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('tanh')(x)

    fake = Dense(1, activation='sigmoid')(x)
    aux = Dense(output_dim, activation='softmax')(x)
    return Model(inputs=[img_input], outputs=[fake, aux])


def generator_containing_discriminator(noise_dim, g, d):
    noise_input = keras.layers.Input(shape=(noise_dim,))
    g_output = g(inputs=[noise_input])
    d.trainable = False
    fake, aux = d(inputs=[g_output])
    return Model(inputs=[noise_input], outputs=[fake, aux])
