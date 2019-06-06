from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate


def create_multi_model(mlp, cnn):
    combined_input = concatenate([mlp.output, cnn.output])

    dense_1 = Dense(4, activation="relu")(combined_input)
    dense_2 = Dense(4, activation="linear")(dense_1)
    model = Model(inputs=[mlp.input, cnn.input], outputs=dense_2)

    optimizer = Adam(lr=3e-4)
    model.compile(loss="mean_absolute_percentage_error", optimizer=optimizer)

    return model
