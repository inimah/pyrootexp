#regression for multidimensional continuous data (multiple classes, each with non-discrete/continuous values)
#using either dat1 or dat2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def createPlots(strErr, n_estimators, train_score, test_score):
   
    plt.figure(figsize=(12,6))
    plt.subplot(1,2,1)
    plt.title('Error Deviance (%s)' % strErr)
    plt.plot(np.arange(n_estimators) + 1, train_score, 'b-', label='Training set')
    plt.plot(np.arange(n_estimators) + 1, test_score, 'r-', label='Test set')
    plt.legend(loc='upper right')
    plt.xlabel('iterations')
    plt.ylabel('Error Deviance(%s)' % strErr)  
    plt.savefig('/home/tita/PyRoot/plots/error_%s.png' % strErr)


    return 0

def createRegressor(X,y):

  
   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

   
   print("X_train shape: %.2d , %.2d" % (X_train.shape[0], X_train.shape[1]))
   print("X_test shape: %.2d , %.2d" % (X_test.shape[0], X_test.shape[1]))
   print("y_train shape: %.2d , %.2d" % (y_train.shape[0], y_train.shape[1]))
   print("y_test shape: %.2d , %.2d" % (y_test.shape[0], y_test.shape[1]))

   params = {'n_estimators': 500, 'max_depth': 5, 'learning_rate':0.01, 'loss': 'ls', 'random_state': 0}


   model = MultiOutputRegressor(GradientBoostingRegressor(**params)).fit(X_train,y_train)
   y_predicted = model.predict(X_test)
   mae = mean_absolute_error(y_test, y_predicted, multioutput='uniform_average')
   mse = mean_squared_error(y_test, y_predicted, multioutput='uniform_average')
   r2 = r2_score(y_test, y_predicted, multioutput='uniform_average')

   maeRaw = mean_absolute_error(y_test, y_predicted, multioutput='raw_values')
   mseRaw = mean_squared_error(y_test, y_predicted, multioutput='raw_values')
   r2Raw = r2_score(y_test, y_predicted, multioutput='raw_values')
   
   c1,c2,c3 = y_test.max(axis=0)
   arrMax = [c1,c2,c3]
   errRelative = (np.abs( y_test - y_predicted ))/y_test
   errRelativeMax = (np.abs( y_test - y_predicted ))/arrMax

   meanErrRelative = errRelative.mean(axis=0)
   meanErrRelativeMax = errRelativeMax.mean(axis=0)

    

   #return X_train, y_train, X_test, y_test, train_score, test_score_mae, test_score_mse, test_score_r2

   return X_train, y_train, X_test, y_test, y_predicted, mae, mse, r2, model


#X_train, y_train, X_test, y_test, y_predicted, mae, mse, r2, model = createRegressor(x,y)
#noncv raw data without normalization / standardization
#mae = 0.52789850276007388
#mse = 0.74901177269174779
#r2 = 0.99286678546963525

#noncv data with min max normalization
#mae = 0.011537672343645014
#mse = 0.00026369855962357533
#r2 = 0.99269978918505808

#noncv data with standardization
#mae = 0.061469116916907041
#mse = 0.0075398897758081817 
#r2 = 0.99239279756115195

##Exact data
#drexact, dphirexact, dzexact
#mae = 0.0083500585421290852
#mse = 0.00017494805356917882
#r2 = 0.9857472928478469
#maeRaw = [ 0.00505656,  0.01056175,  0.00943187]
#mseRaw = [5.43993814e-05,   2.73900167e-04,   1.96544612e-04]
#r2Raw =  [ 0.99206791,  0.96970335,  0.99547061]
#meanErrRelative = [ 0.00698066,  0.03414017,  0.02466064]
#meanErrRelativeMax = [ 0.00520542,  0.01347498,  0.00943187]


def cvRegressor(X,y):

  
   params = {'n_estimators': 500, 'max_depth': 5, 'learning_rate':0.01, 'loss': 'ls', 'random_state': 0}
   model = MultiOutputRegressor(GradientBoostingRegressor(**params))
   y_predicted = cross_val_predict(model, X, y, cv = 10)
   mae = mean_absolute_error(y, y_predicted, multioutput='uniform_average')
   mse = mean_squared_error(y, y_predicted, multioutput='uniform_average')
   r2 = r2_score(y, y_predicted, multioutput='uniform_average')
   
   return y_predicted, mae, mse, r2, model
   
#y_predicted, mae, mse, r2, model = cvRegressor(x,y)


#raw data without normalization / standardization
#mae=2.2590700507815908
#mse=25.600325099243406
#r2=0.81657481348979355

#data with min max normalization
#mae=0.047283691654866074
#mse=0.0066926852775163678
#r2=0.81005142400510366


#data with standardization
#mae=0.24680137203225749
#mse=0.18819118001775281
#r2=0.81180881998224719



    
