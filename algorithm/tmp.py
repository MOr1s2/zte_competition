import json
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE


def preProcess(df):
    df.drop(df.columns[[3, 5, 18, 24, 31, 37, 39, 42, 49,
            59, 65, 71, 80, 87]], axis=1, inplace=True)
    mean = df.mean(axis=0)
    df.fillna(mean, inplace=True)
    return df


def classifyZero():
    train_path = r'/home/liuchang/project/zte/data/train.csv'

    df = pd.read_csv(train_path)
    df = preProcess(df) 
    
    df.loc[df['label'] != 0, 'label'] = 1

    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    et = ExtraTreesClassifier()
    et.fit(X, Y)
    sfm = SelectFromModel(et, prefit=True)
    X = sfm.transform(X)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    ss = StandardScaler()
    ss.fit(X)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes = (256, 64), alpha = 1e-3, max_iter = 500)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    
    return mlp, sfm

def classifyOne2Five():
    train_path = r'/home/liuchang/project/zte/data/train.csv'

    df = pd.read_csv(train_path)
    df = preProcess(df) 
    
    df.drop(df.loc[df['label'] == 0].index, inplace=True)

    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    et = ExtraTreesClassifier()
    et.fit(X, Y)
    sfm = SelectFromModel(et, prefit=True)
    X = sfm.transform(X)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    ss = StandardScaler()
    ss.fit(X)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes = (256, 64), alpha = 5e-3, max_iter = 500)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    print(classification_report(Y_test, Y_pred))

    return mlp, sfm

def classify():
    train_path = r'/home/liuchang/project/zte/data/train.csv'

    df = pd.read_csv(train_path)
    df = preProcess(df) 
    df = shuffle(df, random_state=0)

    del_num = 2200
    del_row = []
    for i, row in df.iterrows():
        if row['label'] == 0 and del_num != 0:
            del_row.append(i)
            del_num -= 1
        if del_num == 0:
            break
    df.drop(del_row, inplace = True)

    X = df.iloc[:, 1:-1]
    Y = df.iloc[:, -1]

    sm = SMOTE(random_state=0)
    X, Y = sm.fit_resample(X, Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

    ss = StandardScaler()
    ss.fit(X)
    X_train = ss.transform(X_train)
    X_test = ss.transform(X_test)

    et = ExtraTreesClassifier()
    et.fit(X, Y)
    sfm = SelectFromModel(et, prefit=True)
    X = sfm.transform(X)
    X_train = sfm.transform(X_train)
    X_test = sfm.transform(X_test)

    mlp = MLPClassifier(hidden_layer_sizes = (1024, 128), alpha = 5e-4, max_iter = 1000)
    mlp.fit(X_train, Y_train)
    Y_pred = mlp.predict(X_test)
    print(classification_report(Y_test, Y_pred))
    
    return mlp, sfm



if __name__ == '__main__':
    test_path = r'/home/liuchang/project/zte/data/test.csv'
    log_path = r'/home/liuchang/project/zte/data/log.txt'
    result_path = r'/home/liuchang/project/zte/data/submit.json'

    test_set = pd.read_csv(test_path)
    test_set = preProcess(test_set)
    X = test_set.iloc[:, 1:]
    ss = StandardScaler()
    x = ss.fit_transform(X)
    # mlp_zero, sfm_zero = classifyZero()
    # mlp_one2five, sfm_one2five = classifyOne2Five()

    # X1 = sfm_zero.transform(X)
    # X2 = sfm_one2five.transform(X)

    # Y1 = mlp_zero.predict(X1)
    # Y2 = mlp_one2five.predict(X2)

    mlp, sfm = classify()
    X = sfm.transform(X)
    Y = mlp.predict(X)

    result = {}
    for i in range(1005):
        # if Y1[i] == 0:
        #     result.update({str(i): int(Y1[i])})
        # else:
        #     result.update({str(i): int(Y2[i])})
        result.update({str(i): int(Y[i])})
    print(pd.value_counts(list(result.values())))
    with open(result_path, 'w') as f:
        f.write(json.dumps(result))
    print("Write Done")