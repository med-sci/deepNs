from tensorflow.keras.layers import (
    Input,
    Add,
    Dense,
    Activation,
    ZeroPadding2D,
    BatchNormalization,
    Flatten,
    Conv2D,
    AveragePooling2D,
    MaxPooling2D,
    GlobalAveragePooling2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.python.keras.layers import concatenate


class InceptionV3:
    def __init__(self, input_shape, classes):
        self._input_shape = input_shape
        self._classes = classes

    @staticmethod
    def conv2d(x, filters, filters_shape, padding='same', strides=(1, 1)):
        x = Conv2D(filters, filters_shape, strides, padding=padding)(x)
        x = BatchNormalization(axis=3, scale=False)(x)
        x = Activation('relu')(x)
        return x

    def inception_block_a(self, x):
        conv1x1a = self.conv2d(x, 64, (1, 1))

        conv1x1b = self.conv2d(x, 48, (1, 1))
        conv3x3a = self.conv2d(conv1x1b, 64, (5, 5))

        conv1x1c = self.conv2d(x, 64, (1, 1))
        conv3x3b = self.conv2d(conv1x1c, 96, (3, 3))
        conv3x3c = self.conv2d(conv3x3b, 96, (3, 3))

        pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = self.conv2d(pool, 32, (1, 1))

        x = concatenate([conv1x1a, conv3x3a, conv3x3c, pool])
        return x

    def inception_block_b(self, x):
        conv1x1a = self.conv2d(x, 192, (1, 1))

        pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = self.conv2d(pool, 192, (1, 1))

        conv1x1b = self.conv2d(x, 128, (1, 1))
        conv1x7a = self.conv2d(conv1x1b, 128, (1, 7))
        conv7x1a = self.conv2d(conv1x7a, 192, (7, 1))

        conv1x1c = self.conv2d(x, 128, (1, 1))
        conv7x1b = self.conv2d(conv1x1c, 128, (7, 1))
        conv1x7b = self.conv2d(conv7x1b, 128, (1, 7))
        conv7x1c = self.conv2d(conv1x7b, 128, (7, 1))
        conv1x7c = self.conv2d(conv7x1c, 192, (1, 7))

        x = concatenate([conv1x1a, pool, conv7x1a, conv1x7c])
        return x

    def inception_block_c(self, x):
        conv1x1a = self.conv2d(x, 320, (1, 1))

        conv1x1b = self.conv2d(x, 384, (1, 1))
        conv1x3a = self.conv2d(conv1x1b, 384, (1, 3))
        conv3x1a = self.conv2d(conv1x1b, 384, (3, 1))
        concat_a = concatenate([conv1x3a, conv3x1a])

        conv1x1b = self.conv2d(x, 448, (1, 1))
        conv3x3 = self.conv2d(conv1x1b, 384, (3, 3))
        conv3x1b = self.conv2d(conv3x3, 384, (3, 1))
        conv1x3b = self.conv2d(conv3x3, 384, (1, 3))
        concat_b = concatenate([conv3x1b, conv1x3b])

        pool = AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool = self.conv2d(pool, 192, (1, 1))
        x = concatenate([conv1x1a, concat_a, concat_b, pool])

        return x



    def reduction_block_a(self, x):
        pool = MaxPooling2D((3, 3), strides=(2, 2))(x)

        conv3x3a = self.conv2d(x, 384, (3, 3), strides=(2 ,2), padding='valid')

        conv1x1 = self.conv2d(x, 64, (1, 1))
        conv3x3b = self.conv2d(conv1x1, 96, (3, 3))
        conv3x3c = self.conv2d(conv3x3b, 96, (3, 3), strides=(2, 2), padding='valid')

        x = concatenate([pool, conv3x3a, conv3x3c])

        return x

    def reduction_block_b(self, x):
        pool = MaxPooling2D((3, 3), (2, 2))(x)

        conv1x1a = self.conv2d(x, 192, (1, 1))
        conv3x3a = self.conv2d(conv1x1a, 320, (3, 3), strides=(2, 2), padding='valid')

        conv1x1b = self.conv2d(x, 192, (1, 1))
        conv1x7 = self.conv2d(conv1x1b, 192, (1, 7))
        conv7x1 = self.conv2d(conv1x7, 192, (7, 1))
        conv3x3b = self.conv2d(conv7x1, 192, (3, 3), padding='valid', strides=(2, 2))

        x = concatenate(pool, conv3x3a, conv3x3b)

        return x

    def build(self):
        x_input = Input(shape=self._input_shape)

        x = self.conv2d(x_input, 32, (3, 3), padding='valid', strides=(2, 2))
        x = self.conv2d(x, 32, (3, 3), padding='valid')
        x = self.conv2d(x, 64, (3, 3), padding='valid')
        x = MaxPooling2D((3,3), strides=(2, 2))(x)

        x = self.conv2d(x, 80, (1, 1), padding='valid')
        x = self.conv2d(x, 192, (3, 3), padding='valid')

        x = MaxPooling2D((3, 3), strides=(2, 2))(x)

        x = self.inception_block_a(x)
        x = self.inception_block_a(x)
        x = self.inception_block_a(x)

        x = self.reduction_block_a(x)

        x = self.inception_block_b(x)
        x = self.inception_block_b(x)
        x = self.inception_block_b(x)
        x = self.inception_block_b(x)

        x = self.reduction_block_b(x)

        x = self.inception_block_c(x)
        x = self.inception_block_c(x)

        x = GlobalAveragePooling2D(name='avg_pool')(x)

        x = Dense(self._classes, activation='softmax', name='predictions')(x)

        model = Model(x_input, x, name='Inception_V3')

        return model


