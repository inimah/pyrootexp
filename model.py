from __future__ import print_function
import sys
import numpy as np
import pandas as pd
from mappingDataCore import *
from sklearn import preprocessing
#from kerasnn.mlp import *
#from kerasnn.mlpRegressor import *
#from kerasnn.multidRegressor import *
from keras.utils import np_utils


'''
call python program by including arguments (3D grid size) and dataset choice. 
python model.py arg1 arg2 arg3 arg4
arg1 = size of grid i
arg2 = size of grid j
arg3 = size of grid m
arg4 = file type (i.e "root", "txt")
arg5 = learning type : "reg" for regression, "cls" for supervised classification
arg6 = option: "1" for cartesian features, "2" for cylindrical/polar features
 

For instance: 
if file to be read is raw ROOT tree file, for regression problem
             python model.py 17 16 18 root reg 1
'''

#filePath = '/home/tita/PyRoot/data/'
#data = extractData(filePath, str(sys.argv[1]), str(sys.argv[2]), str(sys.argv[3]))
#dataset = dataframe.values

#writing to file
#data.to_csv('/home/tita/PyRoot/data/data171618_Reg_b.txt', sep=',',header=False, index=False, index_label=False)

#dataframe = pd.read_csv("/home/tita/PyRoot/data/data171618.txt", header=None)
#dataframe = pd.read_csv("/home/tita/PyRoot/data/data171618_Reg_a.txt", header=None)
#dataframe = pd.read_csv("/home/tita/PyRoot/data/data171618_multidClass_a.txt", header=None)
#dataframe = pd.read_csv("/home/tita/PyRoot/data/data171618_Class_a.txt", header=None)

nClass = int(np.max(y))
y_int = y.astype(int)
#y_nom = np_utils.to_categorical(y_int, nClass+1)

X = dataset[:,0:33]
y = dataset[:,33:36]

#data min max normalization, since NN usually works well with data centralized on [0,1]
def minmaxNormalizedData(X, y):

    

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    X_scaled = (scaler.fit_transform(X))
    y_scaled = (scaler.fit_transform(y))

    return X_scaled, y_scaled

#other normalization techniques using standardization
def standardizedData(dataset):

    X = dataset[:,0:33]
    y = dataset[:,33:36]

    scaler = preprocessing.StandardScaler()
    X_scaled = (scaler.fit_transform(X))
    y_scaled = (scaler.fit_transform(y))

    return X_scaled, y_scaled

dataframe = pd.read_csv("/home/tita/PyRoot/data/data171618_Reg_a.txt", header=None)
dataset = dataframe.values
x, y = minmaxNormalizedData(dataset)
#x, y = standardizedData(dataset)






#X_train, y_train, X_test, y_test, y_predicted, mae, mse, r2, model = createRegressor(data)

#inputDim = 33
#batch = 17
#numClass=data['class'].unique()
#nbClasses = len(numClass)
#nbClasses = 3
#nbEpoch = 200
#nLayer = 3
#nHidden = [306,288,272]
#splitPercentage = 0.2

#createMLP(data, inputDim, nbClasses, batch_size, nbEpoch, nLayer, nHidden, splitPercentage)

#createMLPRegressor(data,inputDim, nbClasses, batch, nbEpoch)

#createStandardMLPRegressor(data,inputDim, nbClasses, batch, nbEpoch)

#createDeepMLPRegressor(data,nbClasses, batch, nbEpoch)
