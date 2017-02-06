import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score



def deep_model():

    model = Sequential()
    model.add(Dense(200, input_dim=33, activation='relu'))
    model.add(Dropout(.2))
    model.add(Activation("linear"))
    model.add(Dense(300, activation='relu'))
    model.add(Activation("linear"))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(18, activation='relu'))
    model.add(Dense(773,activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    #model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def MLPClassifier(x, y, nbClasses, batch, nbEpoch):

    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


    estimators = []
    estimators.append(('mlp', KerasClassifier(build_fn=deep_model, nb_epoch=nbEpoch, batch_size=batch, verbose=0)))
    pipeline = Pipeline(estimators)
    model = pipeline.fit(X_train,y_train)
    y_predicted = model.predict(X_test)
    #accuracy = accuracy_score(y_test, y_predicted)
    #mae = mean_absolute_error(y_test, y_predicted)
    #mse = mean_squared_error(y_test, y_predicted)
    #r2 = r2_score(y_test, y_predicted)
    #acc = model.score(X_test,y_test)

    return model, y_predicted, X_train, X_test, y_train, y_test

#model, y_predicted, X_train, X_test, y_train, y_test = MLPClassifier(X, y_int, 773, 100, 200)

def cvMLPClassifier(x, y, nbClasses, batch, nbEpoch):

       #kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
       #results = cross_val_score(pipeline, X, encoded_Y, cv=kfold)
       #print("Smaller: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

       return 0



