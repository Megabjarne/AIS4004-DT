from sklearn import svm


def train_svm(data: pandas.DataFrame) -> tuple[svm.SVC, svm.SVC]:
    expected_result = data.
    X = [[0, 0], [1, 1]]

    compressor_decay = data[]
    turbine_decay = data[]

    clf = svm.SVC()

    clf.fit(X, y)
