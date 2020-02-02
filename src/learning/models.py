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
    dropout_1 = Dropout(0.3)(dense_1)
    dense_2 = Dense(8, activation="tanh", kernel_regularizer=l2(0.001))(dropout_1)
    dropout_2 = Dropout(0.3)(dense_2)
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
    dropout_1 = Dropout(0.3)(dense_1)
    dense_2 = Dense(6, activation="tanh", kernel_regularizer=l2(0.001))(dropout_1)

    return Model(inputs, dense_2)


def create_cnn(input_shape=(40, 60, 3), filters=(16, 32, 64), regress=False):
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
        conv_2d = Conv2D(conv_filter, (3, 3), activation="tanh", padding="same")(layer_x)
        batch_norm = BatchNormalization(axis=batch_norm_axis)(conv_2d)
        layer_x = MaxPooling2D(pool_size=(2, 2))(batch_norm)

    flatten = Flatten()(layer_x)
    model = Dense(6, activation="tanh", kernel_regularizer=l2(0.001))(flatten)

    if regress:
        model = Dense(6, activation="linear")(model)

    return Model(inputs, model)
