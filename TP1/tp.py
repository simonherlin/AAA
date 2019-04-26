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

def traitment(data):
    print('begin traitment')
    train, test = train_test_split(data)

    x_train = train
    y_train = train['Class']
    del x_train['Class']


    x_test = test
    y_test = test['Class']
    del x_test['Class']


    print('#############')
    print('RandomForestClassifier')
    rdc = RandomForestClassifier()
    rdc.fit(x_train, y_train)
    #print(rdc.feature_importances_)
    #print(rdc.predict(x_test))
    y_pred = rdc.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('#############')
    print('KNeighborsClassifier')
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('#############')
    print('MLPClassifier')
    mpl = MLPClassifier()
    mpl.fit(x_train, y_train)
    y_pred = mpl.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('#############')
    print('tree')
    t = tree.DecisionTreeClassifier()
    t = t.fit(x_train, y_train)
    y_pred = t.predict(x_test)
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    print('#############')
    print('End traitment')


if __name__ == '__main__':
    #first test del
    print('traitement with original data')
    traitment(data)


    print('traitment with class0 remove')
    class0 = data[data.Class == 0]
    class1 = data[data.Class == 1]

    print(len(class0))
    print(len(class1))

    class0 = class0.drop(class0.index[len(class1):(len(class0))])
    print(len(class0))
    new_data = class0.append(class1)
    traitment(new_data)

    print('traitment with up len class1')
    class0 = data[data.Class == 0]
    class1 = class1.append([class1] * int((len(class0) / len(class1))), ignore_index=True)

    new_data = class0.append(class1)
    traitment(new_data)
