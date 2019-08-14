
def create_multi_model(mlp, cnn):
    from keras.layers.core import Dense
    from keras.layers import concatenate
    from keras.models import Model
    from keras.optimizers import Adam

    combined_input = concatenate([mlp.output, cnn.output])

    dense_1 = Dense(4, activation="relu")(combined_input)
    dense_2 = Dense(1, activation="linear")(dense_1)
    model = Model(inputs=[mlp.input, cnn.input], outputs=dense_2)

    optimizer = Adam(lr=3e-4)
    model.compile(loss="mean_squared_error", optimizer=optimizer)

    return model


def create_mlp(input_shape, regress=False):
    from keras.layers import Input
    from keras.layers.core import Dense
    from keras.models import Model

    """More-less copied from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/"""
    inputs = Input(shape=input_shape)
    model = Dense(4, activation="relu")(inputs)

    if regress:
        model = Dense(1, activation="linear")(model)

    return Model(inputs, model)


def create_cnn(input_shape=(60, 40, 3), filters=(16, 32, 64), regress=False):
    from keras.layers.normalization import BatchNormalization
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.layers.core import Activation
    from keras.layers.core import Dense
    from keras.layers.core import Dropout
    from keras.layers import Flatten
    from keras.layers import Input
    from keras.models import Model


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
