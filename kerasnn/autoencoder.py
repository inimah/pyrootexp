from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split


def readData(data, testSplit):
    
    x = data[:,0:33]
    y = data[:,33:36]

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=testSplit,random_state=0)

    return x_train,y_train,x_test,y_test


def create_autoencoder(data, encodingDim, inputShape, x_train,y_train,x_test,y_test):

    encoding_dim = encodingDim

    # this is our input placeholder
    # input shape here is the number of attributes or features we used
    inputData = Input(shape=(inputShape,))

    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='tanh',activity_regularizer=regularizers.activity_l1(10e-5))(inputData)

    # "decoded" is the lossy reconstruction of the input
    decoded = Dense(inputShape, activation='linear')(encoded)

    # this model maps an input to its reconstruction
    autoencoder = Model(input=inputData, output=decoded)

    # this model maps an input to its encoded representation
    encoder = Model(input=inputData, output=encoded)

    # create a placeholder for an encoded (32-dimensional) input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))


    x_train,y_train,x_test,y_test = readData(data, 0.2)

    print("x_train.shape: %s" %(str(x_train.shape)))
    print("y_train.shape: %s" %(str(y_train.shape)))
    print("x_test.shape: %s" %(str(x_test.shape)))
    print("y_test.shape: %s" %(str(y_test.shape)))


    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train, nb_epoch=200, batch_size=33, shuffle=True, validation_data=(x_test, x_test))

    # encode and decode some digits
    # note that we take them from the *test* set
    encoded_feats_prediction = encoder.predict(x_test)
    decoded_feats__prediction = decoder.predict(encoded_feats_prediction)

    return encoded_feats_prediction, decoded_feats__prediction

#encoded_feats_prediction, decoded_feats__prediction = autoencoder_baseline(100, 33, x_train,y_train,x_test,y_test)
