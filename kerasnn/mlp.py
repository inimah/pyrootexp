from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils


def createMLP(data,inputDim, nbClasses, batch, nbEpoch, nLayer, nHidden, splitPercentage):

   X=data[['i', 'j', 'm', 'x_cartesian', 'y_cartesian', 'z', 'mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc']]
   #X=data[['phi', 'r', 'z', 'mCharge', 'd_mr','d_ml','d_ur','d_lr','d_ll','d_ul','d_uc','d_lc','d_fmr','d_fml','d_fmc','d_fur','d_flr','d_fll','d_ful','d_fuc', 'd_flc','d_bmr','d_bml','d_bmc','d_bur','d_blr','d_bll','d_bul','d_buc','d_blc']]
   y=data[['class']]


# splitting data between train and test sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=splitPercentage, random_state=0)
   X_train=X_train.reset_index(drop=True)
   X_test=X_test.reset_index(drop=True)
   Y_train=y_train.reset_index(drop=True)
   Y_test=y_test.reset_index(drop=True)

   X_train_arr = X_train.as_matrix()
   X_test_arr = X_test.as_matrix()
   Y_train_arr = Y_train.as_matrix()
   Y_test_arr = Y_test.as_matrix()

# convert class vectors to binary class matrices
   Y_trainDat = np_utils.to_categorical(Y_train_arr, nbClasses)
   Y_testDat = np_utils.to_categorical(Y_test_arr, nbClasses)


   model = Sequential()
   model.add(Dense(nHidden[0], input_dim=inputDim, init='uniform', activation='relu'))
   for i in range(1,nLayer):     
       model.add(Dropout(0.2))
       model.add(Dense(nHidden[i], activation='relu'))
   model.add(Dense(nbClasses, activation='softmax'))

   model.summary()

   model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

   history = model.fit(X_train_arr, Y_trainDat, batch_size=batch, nb_epoch=nbEpoch, verbose=1, validation_data=(X_test_arr, Y_testDat))
   score = model.evaluate(X_test_arr, Y_testDat, verbose=0)
   print('Test score:', score[0])
   print('Test accuracy:', score[1])




