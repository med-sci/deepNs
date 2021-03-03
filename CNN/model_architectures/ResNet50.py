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
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import glorot_uniform


# noinspection PyMethodFirstArgAssignment
# original implementation coursera CNN course
class ResNet50:
    def __init__(self, input_shape, classes):
        self._input_shape = input_shape
        self._classes = classes

    @staticmethod
    def _identity_block(X, f, filters, stage, block):
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        F1, F2, F3 = filters

        X_shortcut = X

        # First
        X = Conv2D(
            filters=F1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2a",
            kernel_initializer=glorot_uniform(),
        )(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = Activation("relu")(X)

        # Second
        X = Conv2D(
            filters=F2,
            kernel_size=(f, f),
            strides=(1, 1),
            padding="same",
            name=conv_name_base + "2b",
            kernel_initializer=glorot_uniform(),
        )(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
        X = Activation("relu")(X)

        # Third
        X = Conv2D(
            filters=F3,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2c",
            kernel_initializer=glorot_uniform(),
        )(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

        # Final
        X = Add()([X_shortcut, X])
        X = Activation("relu")(X)

        return X

    @staticmethod
    def _convolutional_block(X, f, filters, stage, block, s=2):
        conv_name_base = "res" + str(stage) + block + "_branch"
        bn_name_base = "bn" + str(stage) + block + "_branch"

        F1, F2, F3 = filters

        X_shortcut = X

        # First
        X = Conv2D(
            F1,
            (1, 1),
            strides=(s, s),
            name=conv_name_base + "2a",
            kernel_initializer=glorot_uniform(),
        )(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
        X = Activation("relu")(X)

        # Second
        X = Conv2D(
            F2,
            (f, f),
            strides=(1, 1),
            padding="same",
            name=conv_name_base + "2b",
            kernel_initializer=glorot_uniform(seed=0),
        )(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
        X = Activation("relu")(X)

        # Third
        X = Conv2D(
            F3,
            (1, 1),
            strides=(1, 1),
            padding="valid",
            name=conv_name_base + "2c",
            kernel_initializer=glorot_uniform(seed=0),
        )(X)
        X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

        # Shortcut
        X_shortcut = Conv2D(
            F3,
            (1, 1),
            strides=(s, s),
            padding="valid",
            name=conv_name_base + "1",
            kernel_initializer=glorot_uniform(seed=0),
        )(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name=bn_name_base + "1")(X_shortcut)

        # Final
        X = Add()([X_shortcut, X])
        X = Activation("relu")(X)

        return X

    def build(self):
        X_input = Input(self._input_shape)

        X = ZeroPadding2D((1, 1))(X_input)

        # Stage 1
        X = Conv2D(
            64,
            (7, 7),
            strides=(2, 2),
            name="conv1",
            kernel_initializer=glorot_uniform(),
        )(X)
        X = BatchNormalization(axis=3, name="bn_conv1")(X)
        X = Activation("relu")(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self._convolutional_block(
            X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1
        )
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block="b")
        X = self._identity_block(X, 3, [64, 64, 256], stage=2, block="c")

        # Stage 3
        X = self._convolutional_block(
            X, f=3, filters=[128, 128, 512], stage=3, block="a", s=2
        )
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block="b")
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block="c")
        X = self._identity_block(X, 3, [128, 128, 512], stage=3, block="d")

        # Stage 4
        X = self._convolutional_block(
            X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2
        )
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block="b")
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block="c")
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block="d")
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block="e")
        X = self._identity_block(X, 3, [256, 256, 1024], stage=4, block="f")

        # Stage 5
        X = self._convolutional_block(
            X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2
        )
        X = self._identity_block(X, 3, [512, 512, 2048], stage=5, block="b")
        X = self._identity_block(X, 3, [512, 512, 2048], stage=5, block="c")

        # AVGPOOL
        X = AveragePooling2D((2, 2), name="avg_pool")(X)

        # Dense
        X = Flatten()(X)
        X = Dense(
            self._classes,
            activation="softmax",
            name="fc" + str(self._classes),
            kernel_initializer=glorot_uniform(),
        )(X)

        # Model
        model = Model(inputs=X_input, outputs=X, name="ResNet50")

        return model
