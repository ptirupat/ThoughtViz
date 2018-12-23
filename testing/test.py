import pickle
import random

import keras.backend as K
from PIL import Image
from keras.models import load_model
from keras.utils import to_categorical

from layers.mog_layer import *
from utils.image_utils import *


class Tests():

    def test_deligan_baseline(self, generator_model):
        K.set_learning_phase(False)

        input_noise_dim = 100
        batch_size = 50

        noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))

        random_labels = np.random.randint(0, 10, batch_size)

        conditioned_noise = []
        for i in range(batch_size):
            conditioned_noise.append(np.append(noise[i], to_categorical(random_labels[i], 10)))
        conditioned_noise = np.array(conditioned_noise)

        g = load_model(generator_model, custom_objects={'MoGLayer': MoGLayer})

        # generate images using the generator
        generated_images = g.predict(conditioned_noise, verbose=0)

        image = combine_rgb_images(generated_images)
        image = image * 127.5 + 127.5
        img = Image.fromarray(image.astype(np.uint8))
        img.show()

    def test_deligan_final(self, generator_model, classifier_model, eeg_pkl_file):
        K.set_learning_phase(False)

        # load EEG data
        eeg_data = pickle.load(open(eeg_pkl_file, "rb"), encoding='bytes')
        classifier = load_model(classifier_model)

        x_test = eeg_data[b'x_test']
        y_test = eeg_data[b'y_test']
        y_test = [np.argmax(y) for y in y_test]
        layer_index = 9

        # keras way of getting the output from an intermediate layer
        get_nth_layer_output = K.function([classifier.layers[0].input], [classifier.layers[layer_index].output])

        layer_output = get_nth_layer_output([x_test])[0]

        input_noise_dim = 100
        batch_size = 50

        noise = np.random.uniform(-1, 1, (batch_size, input_noise_dim))

        random_labels = np.random.randint(0, 10, batch_size)

        eeg_feature_vectors = [layer_output[random.choice(np.where(y_test == random_label)[0])] for random_label in random_labels]

        noises, conditionings = [], []
        for i in range(batch_size):
            noises.append(noise[i])
            conditionings.append(eeg_feature_vectors[i])

        g = load_model(generator_model, custom_objects={'MoGLayer': MoGLayer})

        # generate images using the generator
        generated_images = g.predict([np.array(noises), np.array(conditionings)], verbose=0)

        image = combine_rgb_images(generated_images)
        image = image * 127.5 + 127.5
        img = Image.fromarray(image.astype(np.uint8))
        img.show()


if __name__ == '__main__':
    tests = Tests()
    #tests.test_deligan_baseline('../models/gan_models/baseline/deligan/image/generator.model')
    tests.test_deligan_final('../models/gan_models/final/image/generator.model',
                       '../models/eeg_models/image/run_final.h5',
                       '../data/eeg/image/data.pkl')
