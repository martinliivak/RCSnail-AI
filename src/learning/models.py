from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model


def create_mlp(dim, regress=False):
    """More-less copied from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/"""
    model = Sequential()
    model.add(Dense(4, input_dim=dim, activation="relu"))

    if regress:
        model.add(Dense(1, activation="linear"))

    return model


def create_cnn(input_shape=(60, 40, 3), filters=(16, 32, 64), regress=False):
    """More-less copied from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/"""
    batch_norm_axis = -1
    inputs = Input(shape=input_shape)

    x = inputs
    for (layer, conv_filter) in enumerate(filters):
        conv_2d = Conv2D(conv_filter, (3, 3), padding="same")(x)
        relu = Activation("relu")(conv_2d)
        batch_norm = BatchNormalization(axis=batch_norm_axis)(relu)
        x = MaxPooling2D(pool_size=(2, 2))(batch_norm)

    flatten = Flatten()(x)
    dense = Dense(16)(flatten)
    relu_2 = Activation("relu")(dense)
    batch_norm_2 = BatchNormalization(axis=batch_norm_axis)(relu_2)
    dropout = Dropout(0.5)(batch_norm_2)

    # apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    dense_2 = Dense(4)(dropout)
    model = Activation("relu")(dense_2)

    if regress:
        model = Dense(1, activation="linear")(model)

    return Model(inputs, model)
