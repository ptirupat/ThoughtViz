from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential


# define the CNN model for classification
def convolutional_encoder_model(channels, observations, num_classes):
    model = Sequential()
    model.add(BatchNormalization(input_shape=(channels, observations, 1)))
    model.add(Conv2D(32, (1, 4), activation='relu'))
    model.add(Conv2D(25, (channels, 1), activation='relu'))
    model.add(MaxPooling2D((1, 3)))
    model.add(Conv2D(50, (4, 25), activation='relu', data_format='channels_first'))
    model.add(MaxPooling2D((1, 3)))
    model.add(Conv2D(100, (50, 2), activation='relu'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(100, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation='softmax'))
    return model
