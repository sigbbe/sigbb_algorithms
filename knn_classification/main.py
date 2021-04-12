
from math import ceil, sqrt

import numpy as np
import pandas as pd


def encode_labels(df):
    import sklearn.preprocessing
    encoder = {}
    for col in df.columns:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col] = le.transform(df[col])
        encoder[col] = le
    return df, encoder


def train_test_split(X, y, test_size=0.2):
    if (test_size < 0 or test_size > 1):
        raise ValueError("Test size must be a value between 0 and 1")
    return X[int(ceil(test_size * len(X))):], y[0: int(ceil(test_size * len(y)))], X[int(ceil(test_size * len(X))):], y[0: int(ceil(test_size * len(y)))]


def distance(x1, x2):
    """
    “Closeness” is defined in terms of a distance metric, such as Euclidean distance. The
    Euclidean distance between two points or tuples, say, X1 = (x11, x12,..., x1n) and X2 =
    (x21, x22,..., x2n)

    Args:
        x1 (ndarray): an n-dimensional point
        x2 (ndarray): another n-dimensional point
    Returns:
        Eucledian distance between x1 and x2
    """
    return sqrt(pow(np.linalg.norm(x1 - x2), 2))


def main():
    # Load dataset into pandas Dataframe
    mushrooms_df = pd.read_csv('mushrooms.csv')

    # Encoded dataframe and the applied encoder
    mushrooms_encoded_df, encoder = encode_labels(mushrooms_df)

    X_df = mushrooms_encoded_df.drop('class', axis=1)  # attributes
    y_df = mushrooms_encoded_df['class']  # classes

    X_array = X_df.to_numpy()
    y_array = y_df.to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X_array, y_array, 0.33)

    assert len(X_train) == len(y_train)
    assert len(y_train) == 5443
    assert len(X_test) == len(y_test)
    assert len(y_test) == 2681

    indexes = []
    temp_arr = np.ndarray(shape=X_train.shape)
    print(len(temp_arr))
    _indexes = list(map(lambda x: np.where(X_train, x), X_train))
    print(_indexes)
    # while (len(indexes) < len(X_train)):


if __name__ == '__main__':
    main()
