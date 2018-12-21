import os
import pickle

import keras
from keras import optimizers
from keras.callbacks import ModelCheckpoint

from training.models.classification import *


class EEG_Classifier():

    def __init__(self, num_classes, dataset):
        self.num_classes = num_classes
        self.dataset = dataset
        self.eeg_pkl_file = os.path.join('../data/eeg/', self.dataset, 'data.pkl')

    def train(self, model_save_dir, run_id, batch_size, num_epochs):
        
        data = pickle.load(open(self.eeg_pkl_file, 'rb'), encoding='bytes')

        x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

        classifier = convolutional_encoder_model(x_train.shape[1], x_train.shape[2], self.num_classes)

        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)

        # location for the trained model file
        saved_model_file = os.path.join(model_save_dir, str(run_id) + '_final' + '.h5')

        # location for the intermediate model files
        filepath = os.path.join(model_save_dir, str(run_id) + "-model-improvement-{epoch:02d}-{val_acc:.2f}.h5")

        # call back to save model files after each epoch (file saved only when the accuracy of the current epoch is max.)
        callback_checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=False, save_best_only=True, mode='max')

        sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)

        classifier.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        classifier.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size, validation_split=0.25, callbacks=[callback_checkpoint], verbose=False)

        classifier.save(saved_model_file)

        accuracy = classifier.evaluate(x_test, y_test, batch_size=batch_size, verbose=False)
        return accuracy


if __name__ == '__main__':
    batch_size, num_epochs = 128, 100
    digit_classifier = EEG_Classifier(10, 'digit')
    digit_classifier.train('./eeg_digit_classification', 1, batch_size, num_epochs)

    char_classifier = EEG_Classifier(10, 'char')
    char_classifier.train('./eeg_char_classification', 1, batch_size, num_epochs)

    image_classifier = EEG_Classifier(10, 'image')
    image_classifier.train('./eeg_image_classification', 1, batch_size, num_epochs)