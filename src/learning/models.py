# dropout from https://arxiv.org/pdf/1207.0580.pdf
# regularization from https://machinelearningmastery.com/how-to-reduce-overfitting-in-deep-learning-with-weight-regularization/


def time_consistent_loss(y_true, y_pred, coef=0.5):
    import tensorflow.keras.backend as K
    print(y_true.shape)
    print(y_pred.shape)

    return K.mean(K.square(y_pred - y_true), axis=-1) + 0.5 * K.mean(K.square(y_pred[1:] - y_pred[:-1]), axis=-1)


def create_mlp(input_shape=(4,)):
    from tensorflow.keras.layers import Input
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model

    inputs = Input(shape=input_shape)
    dropout_1 = Dropout(rate=0.5)(inputs)
    dense_1 = Dense(10, activation="relu", kernel_regularizer=l2(0.001))(dropout_1)
    dropout_2 = Dropout(rate=0.5)(dense_1)
    dense_2 = Dense(50, activation="relu", kernel_regularizer=l2(0.001))(dropout_2)
    dropout_3 = Dropout(rate=0.5)(dense_2)
    dense_3 = Dense(25, activation="relu", kernel_regularizer=l2(0.001))(dropout_3)
    dropout_4 = Dropout(rate=0.5)(dense_3)
    dense_4 = Dense(10, activation="relu", kernel_regularizer=l2(0.001))(dropout_4)
    dropout_5 = Dropout(rate=0.5)(dense_4)
    return Model(inputs, dropout_5)


def create_nvidia_cnn(input_shape=(40, 60, 3)):
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model

    """ Architecture from https://github.com/tanelp/self-driving-convnet/blob/master/train.py"""
    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(24, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(inputs)
    conv_2 = Convolution2D(36, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(conv_1)
    conv_3 = Convolution2D(48, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(conv_2)
    conv_4 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), padding="same", activation="elu")(conv_3)
    conv_5 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), padding="same", activation="elu")(conv_4)
    flatten = Flatten()(conv_5)
    dense_1 = Dense(1164, kernel_regularizer=l2(0.0005), activation="elu")(flatten)
    dense_2 = Dense(100, kernel_regularizer=l2(0.0005), activation="elu")(dense_1)
    dense_3 = Dense(50, kernel_regularizer=l2(0.0005), activation="elu")(dense_2)
    dense_4 = Dense(10, kernel_regularizer=l2(0.0005), activation="elu")(dense_3)

    return Model(inputs, dense_4)


def create_standalone_nvidia_cnn(activation='linear', input_shape=(40, 60, 3), output_shape=1):
    """
    Activation: linear, softmax.
    Architecture is from nvidia paper mentioned in https://github.com/tanelp/self-driving-convnet/blob/master/train.py
    """
    from tensorflow.keras.layers import Convolution2D
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

    inputs = Input(shape=input_shape)
    conv_1 = Convolution2D(24, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(inputs)
    conv_2 = Convolution2D(36, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(conv_1)
    conv_3 = Convolution2D(48, kernel_size=(5, 5), kernel_regularizer=l2(0.0005), strides=(2, 2), padding="same", activation="elu")(conv_2)
    conv_4 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), padding="same", activation="elu")(conv_3)
    conv_5 = Convolution2D(64, kernel_size=(3, 3), kernel_regularizer=l2(0.0005), padding="same", activation="elu")(conv_4)
    flatten = Flatten()(conv_5)
    dense_1 = Dense(1164, kernel_regularizer=l2(0.0005), activation="elu")(flatten)
    dense_2 = Dense(100, kernel_regularizer=l2(0.0005), activation="elu")(dense_1)
    dense_3 = Dense(50, kernel_regularizer=l2(0.0005), activation="elu")(dense_2)
    dense_4 = Dense(10, kernel_regularizer=l2(0.0005), activation="elu")(dense_3)
    out_dense = Dense(output_shape, activation=activation)(dense_4)

    model = Model(inputs=inputs, outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss=mean_absolute_error, optimizer=optimizer)

    return model


def create_standalone_resnet(activation='linear', input_shape=(40, 60, 3), output_shape=1):
    """
    Activation: linear, softmax.
    Architecture is a pre-trained ResNet50 followed by an MLP
    """
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Flatten
    from tensorflow.keras.layers import Input
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
    from tensorflow.keras.applications import ResNet50

    inputs = Input(shape=input_shape)
    conv_base = ResNet50(weights='imagenet', include_top=False)(inputs)
    conv_base.trainable = False
    flatten = Flatten()(conv_base)

    dense_1 = Dense(2048, kernel_regularizer=l2(0.0005), activation="elu")(flatten)
    dense_2 = Dense(256, kernel_regularizer=l2(0.0005), activation="elu")(dense_1)
    dense_3 = Dense(128, kernel_regularizer=l2(0.0005), activation="elu")(dense_2)
    dense_4 = Dense(10, kernel_regularizer=l2(0.0005), activation="elu")(dense_3)
    out_dense = Dense(output_shape, activation=activation)(dense_4)

    model = Model(inputs=inputs, outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss=mean_absolute_error, optimizer=optimizer)

    return model


def create_multi_model_2(mlp, cnn, output_shape=1):
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

    combined_input = concatenate([cnn.output, mlp.output])
    dense_1 = Dense(20, activation="elu", kernel_regularizer=l2(0.0005))(combined_input)
    dense_2 = Dense(50, activation="elu", kernel_regularizer=l2(0.0005))(dense_1)
    dense_3 = Dense(10, activation="elu", kernel_regularizer=l2(0.0005))(dense_2)
    out_dense = Dense(output_shape, activation="linear")(dense_3)

    model = Model(inputs=[cnn.input, mlp.input], outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss=mean_absolute_error, optimizer=optimizer)

    return model


def create_multi_model_3(mlp, cnn, output_shape=1):
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

    combined_input = concatenate([cnn.output, mlp.output])
    dense_1 = Dense(20, activation="elu", kernel_regularizer=l2(0.0005))(combined_input)
    dropout_1 = Dropout(rate=0.3)(dense_1)
    dense_2 = Dense(50, activation="elu", kernel_regularizer=l2(0.0005))(dropout_1)
    dropout_2 = Dropout(rate=0.3)(dense_2)
    dense_3 = Dense(10, activation="elu", kernel_regularizer=l2(0.0005))(dropout_2)
    dropout_3 = Dropout(rate=0.3)(dense_3)
    out_dense = Dense(output_shape, activation="linear")(dropout_3)

    model = Model(inputs=[cnn.input, mlp.input], outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss=mean_absolute_error, optimizer=optimizer)

    return model


def create_multi_cnn(cnn_list, output_shape=2):
    from tensorflow.keras.layers import concatenate
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.layers import Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.losses import mean_squared_error, mean_absolute_error

    combined_input = concatenate([cnn.output for cnn in cnn_list])
    dense_1 = Dense(150, activation="relu", kernel_regularizer=l2(0.0005))(combined_input)
    dropout_1 = Dropout(rate=0.3)(dense_1)
    dense_2 = Dense(50, activation="relu", kernel_regularizer=l2(0.0005))(dropout_1)
    dropout_2 = Dropout(rate=0.3)(dense_2)
    dense_3 = Dense(10, activation="relu", kernel_regularizer=l2(0.0005))(dropout_2)
    dropout_3 = Dropout(rate=0.3)(dense_3)
    out_dense = Dense(output_shape, activation="linear")(dropout_3)

    model = Model(inputs=[cnn.input for cnn in cnn_list], outputs=out_dense)
    optimizer = Adam(lr=3e-4)
    model.compile(loss=mean_absolute_error, optimizer=optimizer)

    return model
