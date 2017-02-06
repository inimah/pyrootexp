#regression for multidimensional discrete / nominal data (multiple classes, each with discrete values)
#using either dat5 or dat6
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score



def multiClassifier(X, y):

   X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)

   
   print("X_train shape: %.2d , %.2d" % (X_train.shape[0], X_train.shape[1]))
   print("X_test shape: %.2d , %.2d" % (X_test.shape[0], X_test.shape[1]))
   print("y_train shape: %.2d , %.2d" % (y_train.shape[0], y_train.shape[1]))
   print("y_test shape: %.2d , %.2d" % (y_test.shape[0], y_test.shape[1]))


   forest = RandomForestClassifier(n_estimators=100, random_state=1)
   multi_target_forest = MultiOutputClassifier(forest,n_jobs=1).fit(X_train,y_train)
   y_predicted = multi_target_forest.predict(X_test)

   #accuracy = accuracy_score(y_test, y_predicted)
   mae = mean_absolute_error(y_test, y_predicted)
   mse = mean_squared_error(y_test, y_predicted)
   r2 = r2_score(y_test, y_predicted)
   acc = multi_target_forest.score(X_test,y_test)

   #return multi_target_forest, y_predicted, mae, mse, r2
   return multi_target_forest, y_predicted, X_train, X_test, y_train, y_test, mae, mse, r2, acc

#without normalization
#mse = 12.920068027210883
#mae = 0.97721088435374137
#r2 = 0.89810743245669333
#score = multi_target_forest.score(X_test,y_test)
#score (accuracy) = 0.40612244897959182

#with minmax normalization (x)
#mse = 18.078911564625852
#mae = 1.2013605442176871
#r2 = 0.85742283137918029
#acc = 0.38979591836734695

def cvClassifier(X, y):

   forest = RandomForestClassifier(n_estimators=100, random_state=1)
   multi_target_forest = MultiOutputClassifier(forest,n_jobs=1).fit(X_train,y_train)
   y_predicted = cross_val_predict(multi_target_forest, X, y, cv = 10)

   mae = mean_absolute_error(y, y_predicted, multioutput='uniform_average')
   mse = mean_squared_error(y, y_predicted, multioutput='uniform_average')
   r2 = r2_score(y, y_predicted, multioutput='uniform_average')

   return multi_target_forest, y_predicted, mae, mse, r2


#multi_target_forest, y_predicted, mae, mse, r2 = cvClassifier(X, y_int)




