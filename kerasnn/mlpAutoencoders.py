from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor



def deep_model():

   model = Sequential()
   model.add(Dense(306, input_dim=33, init='normal', activation='relu'))
   model.add(Dense(288, init='uniform', activation='relu'))
   model.add(Dense(272, init='uniform', activation='tanh'))
   model.add(Dense(288, init='uniform', activation='tanh'))
   model.add(Dense(306, init='uniform', activation='linear'))
   model.add(Dense(33, init='normal'))

   model.compile(loss='mean_squared_error', optimizer='adam')
   
   

   return model



def createMLPAutoencoders(data,nbClasses, batch, nbEpoch):


   X = data[:,0:33]
   y = data[:,33:36]
   #x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

   seed = 7
   np.random.seed(seed)
   estimators = []
   estimators.append(('standardize', StandardScaler()))
   estimators.append(('mlp', KerasRegressor(build_fn = deep_model, nb_epoch = nbEpoch, batch_size = batch, verbose = 0)))
   model = Pipeline(estimators)
   #model.fit(x_train, x_train, nb_epoch=20, batch_size=16)
   kfold = KFold(n_splits = 10, random_state = seed)
   results = cross_val_score(model, X, X, cv = kfold)
   print("Deep model: %.2f (%.2f) MSE" % (results.mean(), results.std()))


#createMLPAutoencoders(dataset,3, 272, 200)



