import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from collections import Counter


data = pd.read_csv("./creditcard.csv")

#print(data.shape)
#print(data.info)
#print(data.describe)
#print(data.head)
#print(data['Class'].value_count())

train, test = train_test_split(data)

x_train = train
y_train = train['Class']
del x_train['Class']


x_test = test
y_test = test['Class']
del x_test['Class']


print('#############')
rdc = RandomForestClassifier()
rdc.fit(x_train, y_train)
#print(rdc.feature_importances_)
#print(rdc.predict(x_test))
y_pred = rdc.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print('#############')
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print('#############')

mpl = MLPClassifier()
mpl.fit(x_train, y_train)
y_pred = mpl.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

print('#############')

t = tree.DecisionTreeClassifier()
t = t.fit(x_train, y_train)
y_pred = t.predict(x_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

cpt = Counter(data['Class'].values)
print(cpt)
class_cpt  = cpt.values()
a, b  = (cpt[i] for i in cpt)

print('class 0 : ', a)
print('class 1 : ', b)

#first test del

diff = a - b

condition = data.Cl
d = data.where(data.Class == 1 and 
