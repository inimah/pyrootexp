from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasRegressor



def deep_model():

   model = Sequential()
   model.add(Dense(200, input_dim=33, activation='relu'))
   model.add(Dropout(.2))
   model.add(Activation("linear"))
   model.add(Dense(300, activation='relu'))
   model.add(Activation("linear"))
   model.add(Dense(200, activation='relu'))
   model.add(Dense(18, activation='relu'))
   model.add(Dense(3))

   model.compile(loss='mean_squared_error', optimizer='adam')

   return model

def MLPRegressor(x, y, nbClasses, batch, nbEpoch):

   X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

   seed = 3
   np.random.seed(seed)
   estimators = []
   estimators.append(('mlp', KerasRegressor(build_fn = deep_model, nb_epoch = nbEpoch, batch_size = batch, verbose = 0)))
   pipeline = Pipeline(estimators)
   model = pipeline.fit(X_train,y_train)
   y_predicted = model.predict(X_test)
   mae = mean_absolute_error(y_test, y_predicted)
   mse = mean_squared_error(y_test, y_predicted)
   r2 = r2_score(y_test, y_predicted)

   return X_train, X_test, y_train, y_test, model, y_predicted, mae, mse, r2

#model, y_predicted, mae, mse, r2 = MLPRegressor(X, y, 3, 100, 200)

#raw data without normalization / standardization
#mae = 1.0915872166081215
#mse = 3.5036337557073201
#r2 = 0.96984777798937472

#data with min max normalization
#mae = 0.016320052757840129
#mse = 0.00046652429319320865
#r2 = 0.98728341746862858

#data with z-score standardization
#mae = 0.055450888757026341
#mse = 0.0062343208688496994
#r2 = 0.99367991989563098


##Exact
#drexact, dphirexact, dzexact
#X_train, X_test, y_train, y_test, model, y_predicted, mae, mse, r2 = MLPRegressor(Xnorm, ynorm, 3, 100, 200)
#mae=0.01105490264907011
#mse = 0.00026797110870795934
#r2 = 0.9864414452505933
#maeRaw = [ 0.00818956,  0.0079235 ,  0.01705165]
#mseRaw = [ 0.00014904,  0.00016362,  0.00049125]
#r2Raw = [ 0.97826779,  0.98190134,  0.98867916]
#meanErrRelative[ 0.01225635,  0.02299114,  0.03194341]
#meanErrRelativeMax[ 0.00843065,  0.01010903,  0.01705165]

def cvMLPRegressor(x, y, nbClasses, batch, nbEpoch):

   seed = 3
   np.random.seed(seed)
   estimators = []
   estimators.append(('mlp', KerasRegressor(build_fn = deep_model, nb_epoch = nbEpoch, batch_size = batch, verbose = 0)))
   model = Pipeline(estimators)
   kfold = KFold(n_splits = 10, random_state = seed)
   y_predicted = cross_val_predict(model, x, y, cv = kfold)
   mae = mean_absolute_error(y, y_predicted)
   mse = mean_squared_error(y, y_predicted)
   r2 = r2_score(y, y_predicted)

   return model, y_predicted, mae, mse, r2

#model, y_predicted2, mae_2, mse_2, r2_2 = cvMLPRegressor(x, y, 3, 100, 200)

#raw data without normalization / standardization
#mae = 2.7303752708257729
#mse = 53.591261330236385
#r2 = 0.55235105912103799

#data with min max normalization
#mae = 0.054054063472151088
#mse = 0.012379132300838473
#r2 = 0.6667911869150045

#data with z-score standardization
#mae = 0.28767800526290127
#mse = 0.33916596777679858
#r2 = 0.66083403222320147


# input features is a corrupted version of x (autoencoder results of x)
def MLPAutoencoders(x_train_encoded, x_train_decoded, x_test_encoded, x_test_decoded, x_train_ae, x_test_ae, x_train, y_train, x_test, y_test, nbClasses, batch, nbEpoch):

   estimators = []
   estimators.append(('mlp', KerasRegressor(build_fn = deep_model, nb_epoch = nbEpoch, batch_size = batch, verbose = 0)))
   pipeline = Pipeline(estimators)

   #model1 = model is fitted on original x features
   model1 = pipeline.fit(x_train,y_train)

   #model2 = model is fitted on corrupted version of x (encoded of x)
   model2 = pipeline.fit(x_train_decoded,y_train)

   #model3 = model is fitted on full autoencoders version of x 
   model3 = pipeline.fit(x_train_ae,y_train)

   #predicting model 1 results on test data
   y_predicted1 = model1.predict(x_test)
   mae1 = mean_absolute_error(y_test, y_predicted1)
   mse1 = mean_squared_error(y_test, y_predicted1)
   r21 = r2_score(y_test, y_predicted1)

   #predicting model 2 results on test data (encoded)
   y_predicted2 = model2.predict(x_test_decoded)
   mae2 = mean_absolute_error(y_test, y_predicted2)
   mse2 = mean_squared_error(y_test, y_predicted2)
   r22 = r2_score(y_test, y_predicted2)
   
   #predicting model 3 results on test data (decoded)
   y_predicted3 = model3.predict(x_test_ae)
   mae3 = mean_absolute_error(y_test, y_predicted3)
   mse3 = mean_squared_error(y_test, y_predicted3)
   r23 = r2_score(y_test, y_predicted3)
   

   return model1, model2, model3, y_predicted1, y_predicted2, y_predicted3, mae1, mae2, mae3, mse1, mse2, mse3, r21, r22, r23

#model1, model2, model3, y_predicted1, y_predicted2, y_predicted3, mae1, mae2, mae3, mse1, mse2, mse3, r21, r22, r23 = MLPAutoencoders(x_train_encoded, x_train_decoded, x_test_encoded, x_test_decoded, x_train_ae, x_test_ae, x_train, y_train, x_test, y_test, 3, 100, 200)

#data with min max normalization
#mae1 = 0.020975078582091921
#mse1 = 0.00083366622969472065
#r21 = 0.97727581271070108

#mae2 = 0.18146567360567745
#mse2 = 0.054060022956051473
#r22 = -0.47357544633535348

#mae3 = 0.018121573066966191
#mse3 = 0.00064991699848163879
#r23 = 0.98228447420569764

#similar model with cross validation method instead of splitting data
def cvMLPAutoencoders(x_encoded, x_decoded, x_ae, x, y, nbClasses, batch, nbEpoch):

   estimators = []
   estimators.append(('standardize', StandardScaler()))
   estimators.append(('mlp', KerasRegressor(build_fn = deep_model, nb_epoch = nbEpoch, batch_size = batch, verbose = 0)))
   model = Pipeline(estimators)
   kfold = KFold(n_splits = 10, random_state = 7)
   
   #using original data
   y_predicted1 = cross_val_predict(model, x, y, cv = kfold)
   mae1 = mean_absolute_error(y, y_predicted1)
   mse1 = mean_squared_error(y, y_predicted1)
   r21 = r2_score(y, y_predicted1)

   #using corrupted data (separated encoder decoder)
   y_predicted2 = cross_val_predict(model, x_decoded, y, cv = kfold)
   mae2 = mean_absolute_error(y, y_predicted2)
   mse2 = mean_squared_error(y, y_predicted2)
   r22 = r2_score(y, y_predicted2)

   #using autoencoder corrupted data
   y_predicted3 = cross_val_predict(model, x_ae, y, cv = kfold)
   mae3 = mean_absolute_error(y, y_predicted3)
   mse3 = mean_squared_error(y, y_predicted3)
   r23 = r2_score(y, y_predicted3)


   return model, y_predicted1, y_predicted2, y_predicted3, mae1, mae2, mae3, mse1, mse2, mse3, r21, r22, r23

#model, y_predicted1, y_predicted2, y_predicted3, mae1, mae2, mae3, mse1, mse2, mse3, r21, r22, r23 = cvMLPAutoencoders(x_encoded, x_decoded, x_ae, x, y, 3, 100, 200)

#raw data without normalization / standardization
#mae =
#mse =
#r2 =

#data with min max normalization
#mae1 = 0.052616118409197062
#mse1 = 0.011903346840854062
#r21 = 0.67959789295479411

#mae2 = 0.056028912511401942
#mse2 = 0.011482590536279884
#r22 = 0.69092338051224556

#mae3 = 0.056162803563851123
#mse3 = 0.013644278530364715
#r23 = 0.63273727560081305


#data with z-score standardization



#pipeline, results_ori, results_encoded, results_decoded = cvMLPAutoencoders(encoded_predict, decoded_predict, x, y, 3, 272, 200)

#Deep model ori: 31.64 (58.61) MSE
#Deep model encoded: 50.80 (78.89) MSE
#Deep model decoded: 168.65 (115.82) MSE

