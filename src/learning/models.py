from keras import regularizers
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.models import load_model

def create_slim_model(input_shape=(64, 64, 3)):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal", activation='relu', input_shape=input_shape,
                     kernel_regularizer=regularizers.l2(0.1),
                     activity_regularizer=regularizers.l1(0.1)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), kernel_initializer="he_normal", activation='relu',
                     kernel_regularizer=regularizers.l2(0.1),
                     activity_regularizer=regularizers.l1(0.1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1, activation='linear',
                    kernel_regularizer=regularizers.l2(0.1),
                    activity_regularizer=regularizers.l1(0.1)))

    model.compile(loss="mse", optimizer="adam")
    return model
