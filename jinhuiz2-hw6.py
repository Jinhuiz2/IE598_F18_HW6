from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import cross_val_score



#From Iris dataset get iris data, split it with 90% training and 10% test
iris = datasets.load_iris()
X, y = iris.data, iris.target



in_sample = []
out_sample = []
for k in range(1,11):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,
                                                    random_state=k)
    scaler = preprocessing.StandardScaler().fit(X_train)
    dt = DecisionTreeClassifier(max_depth = 6, criterion = 'gini', random_state = 1)
    dt.fit(X_train, y_train)
    y_pred_out = dt.predict(X_test)
    y_pred_in = dt.predict(X_train)
    out_sample_score = accuracy_score(y_test, y_pred_out)
    in_sample_score = accuracy_score(y_train, y_pred_in)
    in_sample.append(in_sample_score)
    out_sample.append(out_sample_score)
    print('Random State: %d, in_sample: %.3f, out_sample: %.3f'%(k, in_sample_score,
                                                             out_sample_score))
    
    
in_sample_mean = np.mean(in_sample_score)
in_sample_std = np.std(in_sample_score)
out_sample_mean = np.mean(out_sample_score)
out_sample_std = np.std(out_sample_score)
print('In sample mean: %.f' %in_sample_mean)
print('In sample standard deviation: %.f' %in_sample_std)
print('Out sample mean: %.f' %out_sample_mean)
print('Out sample standard deviation: %.f' %out_sample_std)
print('\n')



in_sample = []
out_sample = []

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.1,
                                                random_state=k)
scaler = preprocessing.StandardScaler().fit(X_train)
dt = DecisionTreeClassifier(max_depth = 6, criterion = 'gini', random_state = 1)
dt.fit(X_train, y_train)
y_pred_out = dt.predict(X_test)
y_pred_in = dt.predict(X_train)
in_sample_score = cross_val_score(dt, X_train, y_train, cv=10)
out_sample_score = accuracy_score(y_test, y_pred_out)
in_sample.append(in_sample_score)
out_sample.append(out_sample_score)
print('In sample CV score for every fold:')
for i in in_sample_score:
    print(i)
print('Mean of sample CV score: ', np.mean(in_sample_score))
print('Standard deviation of sample CV score: ', np.std(in_sample_score))
print('\n')
    
print("My name is {Jinhui Zhang}")
print("My NetID is: {jinhuiz2}")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")



