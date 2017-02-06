from keras.layers import *
from keras.models import Model
from keras.datasets import mnist
import numpy as np
from keras import regularizers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def splitData(x, y, testSplit):
    
    
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=testSplit,random_state=0)

    return x_train,y_train,x_test,y_test


def plotAutoencoders(x, x_decoded):
    
    colors = {0: 'b', 1: 'g', 2: 'r', 3:'c', 4:'m', 5:'y', 6:'k', 7:'orange', 8:'darkgreen', 9:'maroon'}

    markers = {0: 'o', 1: '+', 2: 'v', 3:'<', 4:'>', 5:'^', 6:'s', 7:'p', 8:'*', 9:'x'}

    plt.figure(figsize=(10, 10))
    
    #plotting r and dr
    plt.scatter(x[:, 0], y[:, 0], c=colors[0], marker=markers[0])
    plt.scatter(x_decoded[:, 0], y[:, 0], c=colors[7], marker=markers[5])
    plt.savefig('/home/tita/PyRoot/plots/scatter_encoded_r.png')

    #plotting phi and dphi
    plt.scatter(x[:, 2], y[:, 1], c=colors[0], marker=markers[0])
    plt.scatter(x_decoded[:, 2], y[:, 1], c=colors[7], marker=markers[5])
    plt.savefig('/home/tita/PyRoot/plots/scatter_encoded_phi.png')
    
    #plotting z and dz
    plt.scatter(x[:, 1], y[:, 2], c=colors[0], marker=markers[0])
    plt.scatter(x_decoded[:, 1], y[:, 2], c=colors[7], marker=markers[5])
    plt.savefig('/home/tita/PyRoot/plots/scatter_encoded_z.png')

    #plotting mlpautoencoders
    plt.scatter(x_test_decoded[:, 0], y_test[:, 0], c=colors[0], marker=markers[0])
    plt.scatter(x_test_decoded[:, 0], y_predicted2[:, 0], c=colors[7], marker=markers[5])
    plt.savefig('/home/tita/PyRoot/plots/scatter_mlpautoencoder.png')
   
    return 0


#creating autoencoders with splitted data
def deepAutoencoders(x, y, encodingDim, inputShape):

    encoding_dim = encodingDim

    inputDat = Input(shape=(inputShape,))
    encoded = Dense(300, activation='relu')(inputDat)
    encoded = Dense(200, activation='relu')(encoded)
    encoded = Dense(300, activation='relu')(encoded)
    decoded = Dense(200, activation='relu')(encoded)
    decoded = Dense(300, activation='relu')(decoded)
    decoded = Dense(inputShape, activation='sigmoid')(decoded)

    # maps an input to its reconstruction
    autoencoder = Model(input=inputDat, output=decoded)

    # maps an input to its encoded representation
    encoder = Model(input=inputDat, output=encoded)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='mse')

    x_train,y_train,x_test,y_test,x,y = splitData(x, y, 0.2)

    autoencoder.fit(x_train, x_train, nb_epoch=100, batch_size=100, shuffle=True, validation_data=(x_test, x_test))

    x_train_encoded = encoder.predict(x_train)
    x_train_decoded = decoder.predict(x_train_encoded)

    x_test_encoded = encoder.predict(x_test)
    x_test_decoded = decoder.predict(x_test_encoded)

    #full corrupted version using autoencoder model instead of separated encoder - decoder
    x_train_ae = autoencoder.predict(x_train)
    x_test_ae = autoencoder.predict(x_test)

   
    return autoencoder, encoder, decoder, x_train_encoded, x_train_decoded, x_test_encoded, x_test_decoded, x_train_ae, x_test_ae, x_train, y_train, x_test, y_test
    
#autoencoder, encoder, decoder, x_train_encoded, x_train_decoded, x_test_encoded, x_test_decoded, x_train_ae, x_test_ae, x_train, y_train, x_test, y_test = deepAutoencoders(x, y, 300, 33)

#raw data without normalization / standardization

#data with min max normalization
'''
Epoch 1/100
3916/3916 [==============================] - 0s - loss: 0.0556 - val_loss: 0.0118
Epoch 2/100
3916/3916 [==============================] - 0s - loss: 0.0070 - val_loss: 0.0036
Epoch 3/100
3916/3916 [==============================] - 0s - loss: 0.0026 - val_loss: 0.0016
Epoch 4/100
3916/3916 [==============================] - 0s - loss: 0.0014 - val_loss: 0.0012
Epoch 5/100
3916/3916 [==============================] - 0s - loss: 9.2187e-04 - val_loss: 7.6463e-04

..
Epoch 95/100
3916/3916 [==============================] - 1s - loss: 4.8796e-05 - val_loss: 4.8338e-05
Epoch 96/100
3916/3916 [==============================] - 1s - loss: 3.1180e-05 - val_loss: 1.1737e-04
Epoch 97/100
3916/3916 [==============================] - 1s - loss: 7.7038e-05 - val_loss: 6.9926e-05
Epoch 98/100
3916/3916 [==============================] - 1s - loss: 4.6670e-05 - val_loss: 7.6843e-05
Epoch 99/100
3916/3916 [==============================] - 1s - loss: 5.5404e-05 - val_loss: 4.3900e-05
Epoch 100/100
3916/3916 [==============================] - 1s - loss: 4.8584e-05 - val_loss: 8.3690e-05
'''

#data with z-score standardization
#mae =
#mse =
#r2 =


#autoencoders for cv case
def cvAutoencoders(x, y, encodingDim, inputShape):

    encoding_dim = encodingDim

    inputDat = Input(shape=(inputShape,))
    encoded = Dense(300, activation='relu')(inputDat)
    encoded = Dense(200, activation='relu')(encoded)
    encoded = Dense(300, activation='relu')(encoded)
    decoded = Dense(200, activation='relu')(encoded)
    decoded = Dense(300, activation='relu')(decoded)
    decoded = Dense(inputShape, activation='sigmoid')(decoded)

    # maps an input to its reconstruction
    autoencoder = Model(input=inputDat, output=decoded)

    # maps an input to its encoded representation
    encoder = Model(input=inputDat, output=encoded)

    # create a placeholder for an encoded input
    encoded_input = Input(shape=(encoding_dim,))
    # retrieve the last layer of the autoencoder model
    decoder_layer = autoencoder.layers[-1]
    # create the decoder model
    decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(x, x, nb_epoch=200, batch_size=100, shuffle=True)

    #corrupted version of x
    x_encoded = encoder.predict(x)
    x_decoded = decoder.predict(x_encoded)

    #full corrupted version using autoencoder model instead of separated encoder - decoder
    x_ae = autoencoder.predict(x)
    
    return autoencoder, encoder, decoder, x_encoded, x_decoded, x_ae


#autoencoder, encoder, decoder, x_encoded, x_decoded, x_ae = cvAutoencoders(x, y, 300, 33)

#raw data without normalization / standardization

#data with min max normalization

'''
Epoch 1/200
4896/4896 [==============================] - 1s - loss: 0.0463     
Epoch 2/200
4896/4896 [==============================] - 0s - loss: 0.0044     
Epoch 3/200
4896/4896 [==============================] - 0s - loss: 0.0014     
Epoch 4/200
4896/4896 [==============================] - 0s - loss: 8.2284e-04     
Epoch 5/200
4896/4896 [==============================] - 0s - loss: 7.0078e-04     
Epoch 6/200
4896/4896 [==============================] - 0s - loss: 5.2365e-04     
Epoch 7/200
4896/4896 [==============================] - 0s - loss: 4.9940e-04     
Epoch 8/200
4896/4896 [==============================] - 0s - loss: 4.3658e-04     
Epoch 9/200
4896/4896 [==============================] - 0s - loss: 4.1808e-04     
Epoch 10/200
4896/4896 [==============================] - 0s - loss: 2.9041e-04   

..

Epoch 190/200
4896/4896 [==============================] - 1s - loss: 2.8009e-05     
Epoch 191/200
4896/4896 [==============================] - 1s - loss: 1.7387e-05     
Epoch 192/200
4896/4896 [==============================] - 1s - loss: 1.3447e-05     
Epoch 193/200
4896/4896 [==============================] - 1s - loss: 1.4688e-05     
Epoch 194/200
4896/4896 [==============================] - 1s - loss: 2.3136e-05     
Epoch 195/200
4896/4896 [==============================] - 1s - loss: 6.4743e-05     
Epoch 196/200
4896/4896 [==============================] - 1s - loss: 7.0092e-04     
Epoch 197/200
4896/4896 [==============================] - 1s - loss: 0.0055     
Epoch 198/200
4896/4896 [==============================] - 1s - loss: 3.7785e-04     
Epoch 199/200
4896/4896 [==============================] - 1s - loss: 1.5515e-04     
Epoch 200/200
4896/4896 [==============================] - 1s - loss: 1.0595e-04     
'''
#data with z-score standardization



def apiAutoencoder(X_train,X_test):

    # input shape: (nb_samples, 32)
    encoder = containers.Sequential([Dense(17, input_dim=33), Dense(33)])
    decoder = containers.Sequential([Dense(18, input_dim=33), Dense(33)])

    autoencoder = AutoEncoder(encoder=encoder, decoder=decoder, output_reconstruction=True)
    model = Sequential()
    model.add(autoencoder)

    # training the autoencoder:
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X_train, X_train, nb_epoch=10)

    # predicting compressed representations of inputs:
    autoencoder.output_reconstruction = False  # the model has to be recompiled after modifying this property
    model.compile(optimizer='sgd', loss='mse')
    representations = model.predict(X_test)

    # the model is still trainable, although it now expects compressed representations as targets:
    model.fit(X_test, representations, nb_epoch=1)  # in this case the loss will be 0, so it's useless

    # to keep training against the original inputs, just switch back output_reconstruction to True:
    autoencoder.output_reconstruction = True
    model.compile(optimizer='sgd', loss='mse')
    model.fit(X_train, X_train, nb_epoch=10)

    return model
