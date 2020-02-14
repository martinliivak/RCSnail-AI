# dropout from https://arxiv.org/pdf/1207.0580.pdf
# regularization from https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/


def create_multi_model(mlp, cnn, output_shape=4):
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    combined_input = concatenate([cnn.output, mlp.output])

    dense_1 = Dense(12, activation="tanh", kernel_regularizer=l2(0.001))(combined_input)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(8, activation="tanh", kernel_regularizer=l2(0.001))(dropout_1)
    dropout_2 = Dropout(0.5)(dense_2)
    out_dense = Dense(output_shape, activation="linear")(dropout_2)

    model = Model(inputs=[cnn.input, mlp.input], outputs=out_dense)
    optimizer = Adam(lr=3e-4)

    # MAE usage from https://arxiv.org/abs/1809.04843
    model.compile(loss="mean_absolute_error", optimizer=optimizer)

    return model


def create_mlp(input_shape=(4,)):
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    """More-less copied from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/"""
    inputs = Input(shape=input_shape)
    dense_1 = Dense(8, activation="tanh", kernel_regularizer=l2(0.001))(inputs)
    dropout_1 = Dropout(0.5)(dense_1)
    dense_2 = Dense(6, activation="tanh", kernel_regularizer=l2(0.001))(dropout_1)

    return Model(inputs, dense_2)


def create_cnn(input_shape=(40, 60, 3), filters=(16, 32, 64), kernel=(3, 3), regress=False):
    from tensorflow.keras.layers import BatchNormalization
    from tensorflow.keras.layers import Conv2D
    from tensorflow.keras.layers import MaxPooling2D
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.regularizers import l2

    """More-less copied from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/"""
    batch_norm_axis = -1
    inputs = Input(shape=input_shape)

    layer_x = inputs
    for (layer, conv_filter) in enumerate(filters):
        conv_2d = Conv2D(conv_filter, kernel, activation="tanh", padding="same")(layer_x)
        batch_norm = BatchNormalization(axis=batch_norm_axis)(conv_2d)
        layer_x = MaxPooling2D(pool_size=(2, 2))(batch_norm)

    flatten = Flatten()(layer_x)
    model = Dense(6, activation="tanh", kernel_regularizer=l2(0.001))(flatten)

    if regress:
        model = Dense(6, activation="linear")(model)

    return Model(inputs, model)


def create_mlp_2(input_shape=(4,)):
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    dropout_1 = Dropout(0.5)(inputs)
    dense_1 = Dense(20, activation="relu", kernel_regularizer=l2(0.001))(dropout_1)
    dropout_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(10, activation="relu", kernel_regularizer=l2(0.001))(dropout_2)

    return Model(inputs, dense_2)


def create_cnn_2(input_shape=(40, 60, 3)):
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    """ Architecture from https://github.com/tanelp/self-driving-convnet/blob/master/train.py"""
    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(24, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(inputs)
    conv_2 = Convolution2D(36, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_1)
    conv_3 = Convolution2D(48, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_2)
    conv_4 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_3)
    conv_5 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_4)
    conv_6 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_5)
    flatten = Flatten()(conv_6)
    dense_1 = Dense(1164, kernel_regularizer=l2(0.0005), activation="relu")(flatten)
    dense_2 = Dense(100, kernel_regularizer=l2(0.0005), activation="relu")(dense_1)
    dense_3 = Dense(50, kernel_regularizer=l2(0.0005), activation="relu")(dense_2)
    dense_4 = Dense(10, kernel_regularizer=l2(0.0005), activation="relu")(dense_3)

    return Model(inputs, dense_4)


def create_cnn_alone(input_shape=(40, 60, 3), output_shape=1):
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    """ Architecture from nvidia paper mentioned in https://github.com/tanelp/self-driving-convnet/blob/master/train.py"""
    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(24, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(inputs)
    conv_2 = Convolution2D(36, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_1)
    conv_3 = Convolution2D(48, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_2)
    conv_4 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_3)
    conv_5 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_4)
    conv_6 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="relu")(conv_5)
    flatten = Flatten()(conv_6)
    dense_1 = Dense(1164, kernel_regularizer=l2(0.0005), activation="relu")(flatten)
    dense_2 = Dense(100, kernel_regularizer=l2(0.0005), activation="relu")(dense_1)
    dense_3 = Dense(50, kernel_regularizer=l2(0.0005), activation="relu")(dense_2)
    dense_4 = Dense(10, kernel_regularizer=l2(0.0005), activation="relu")(dense_3)
    out_dense = Dense(output_shape, activation="linear")(dense_4)

    model = Model(inputs=inputs, outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss="mae", optimizer=optimizer)

    return model


def create_multi_model_2(mlp, cnn, output_shape=1):
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    combined_input = concatenate([cnn.output, mlp.output])

    dense_1 = Dense(20, activation="relu", kernel_regularizer=l2(0.001))(combined_input)
    dense_2 = Dense(10, activation="relu", kernel_regularizer=l2(0.001))(dense_1)
    out_dense = Dense(output_shape, activation="linear")(dense_2)

    model = Model(inputs=[cnn.input, mlp.input], outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss="mae", optimizer=optimizer)

    return model
