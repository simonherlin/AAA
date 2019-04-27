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
from sklearn.utils import shuffle


data = pd.read_csv("./creditcard.csv")

#print(data.shape)
#print(data.info)
#print(data.describe)
#print(data.head)
#print(data['Class'].value_count())

def traitment(x_train, y_train, x_test, y_test):
    print('begin traitment')

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
    print('begin traitment')
    train, test = train_test_split(data)

    class0_master = train[train.Class == 0]
    class1_master = train[train.Class == 1]

    train = shuffle(train)
    x_train = train
    y_train = train['Class']
    del x_train['Class']

    x_test = test
    y_test = test['Class']
    del x_test['Class']

    print('traitement with original data')
    traitment(x_train, y_train, x_test, y_test)


    print('traitment with class0 remove')
    class0 = class0_master
    class1 = class1_master
    class0 = class0.drop(class0.index[len(class1):(len(class0))])
    #print(len(class0))
    class0 = class0.append(class1)

    new_value = shuffle(class0)
    x_train = new_value
    y_train = new_value['Class']
    del x_train['Class']

    traitment(x_train, y_train, x_test, y_test)

    print('traitment with up len class1')

    class0 = class0_master
    class1 = class1_master

    class1 = class1.append([class1] * int((len(class0) / len(class1))), ignore_index=True)

    class0 = class0.append(class1)

    new_value = shuffle(class0)
    x_train = new_value
    y_train = new_value['Class']
    del x_train['Class']

    traitment(x_train, y_train, x_test, y_test)
