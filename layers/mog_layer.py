from keras.engine.topology import Layer
from keras.layers import regularizers, initializers
import keras.backend as K


class MoGLayer(Layer):

    def __init__(self,
                 kernel_regularizer=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(MoGLayer, self).__init__(**kwargs)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.std = self.add_weight(shape=(input_dim,),
                                      name='std',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)

        self.mean = self.add_weight(shape=(input_dim,),
                                    initializer=self.bias_initializer,
                                    name='mean')

        self.built = True

    def call(self, inputs):
        output = inputs * self.std
        output = K.bias_add(output, self.mean)
        return output

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = input_shape[-1]
        return tuple(output_shape)
