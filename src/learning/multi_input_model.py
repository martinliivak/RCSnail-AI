from keras import regularizers
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.models import load_model

from sklearn.model_selection import train_test_split
from keras.layers.core import Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import concatenate
import numpy as np
import argparse
import locale
import os

# Currently raw copy from https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/
models = None
trainAttrX = None
trainImagesX = None
trainY = None

testAttrX = None
testImagesX = None
testY = None

# create the MLP and CNN models
mlp = models.create_mlp(trainAttrX.shape[1], regress=False)
cnn = models.create_cnn(64, 64, 3, regress=False)

# create the input to our final set of layers as the *output* of both
# the MLP and CNN
combinedInput = concatenate([mlp.output, cnn.output])

# our final FC layer head will have two dense layers, the final one
# being our regression head
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

# our final model will accept categorical/numerical data on the MLP
# input and images on the CNN input, outputting a single value (the
# predicted price of the house)
model = Model(inputs=[mlp.input, cnn.input], outputs=x)

opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(
    [trainAttrX, trainImagesX], trainY,
    validation_data=([testAttrX, testImagesX], testY),
    epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting house prices...")
preds = model.predict([testAttrX, testImagesX])

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
