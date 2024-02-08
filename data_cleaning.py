import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def Standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def process_data(dataset):
    dataset = pd.read_csv(dataset)
    X = dataset.iloc[:,1:]
    y = dataset['HomeGoals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.3)
    X_train, X_test = Standardize(X_train, X_test)

    return X_train, X_test, y_train, y_test