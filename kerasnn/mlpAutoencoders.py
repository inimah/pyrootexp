from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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
   model.add(Dense(272, init='uniform', activation='relu'))
   model.add(Dense(288, init='uniform', activation='relu'))
   model.add(Dense(306, init='uniform', activation='relu'))
   model.add(Dense(3, init='normal'))

   model.compile(loss='mean_squared_error', optimizer='adam')
   
   

   return model



def createMLPAutoencoders(Xtrain, ytrain, Xtest, ytest, batch, nbEpoch):


   estimators = []
   estimators.append(('mlp', KerasRegressor(build_fn = deep_model, nb_epoch = nbEpoch, batch_size = batch, verbose = 0)))
   pipeline = Pipeline(estimators)
   model = pipeline.fit(Xtrain,ytrain)
   y_predicted = model.predict(Xtest)
   mae = mean_absolute_error(ytest, y_predicted)
   mse = mean_squared_error(ytest, y_predicted)
   r2 = r2_score(y_test, y_predicted)

   #kfold = KFold(n_splits = 10, random_state = seed)
   #results = cross_val_score(model, Xtest, ytest, cv = 10)
   #print("Deep model: %.2f (%.2f) MSE" % (results.mean(), results.std()))

   return model, y_predicted, mae, mse, r2


#model, y_predicted, mae, mse, r2 = createMLPAutoencoders(x_train_encoded, y_train, x_test_encoded, y_test,100, 200)



