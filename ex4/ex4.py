import numpy as np
import tensorflow as tf

def load_data(file1, file2):
    return np.loadtxt(file1), np.loadtxt(file2)

def data_fn():
    X, y = load_data("../data/ex3data1_X.txt", "../data/ex3data1_y.txt")
    assert X.shape[0] == y.shape[0]

    out = {}
    for j in range(400):
        tmp = [None] * 5000
        for i in range(5000):
            tmp[i] = X[i][j]
        out["px"+str(j)] = tmp

    outY = [None] * 5000
    for i in range(5000):
        outY[i] = int(y[i]%10)

    dataset = tf.data.Dataset.from_tensor_slices((dict(out), outY))
    dataset = dataset.batch(5000)
    return dataset.make_one_shot_iterator().get_next()


feat = [None] * 400
for i in range(400):
    feat[i] = tf.feature_column.numeric_column("px"+str(i))

estimator = tf.estimator.DNNClassifier(feature_columns=feat, hidden_units=[25], n_classes=10)
estimator.train(input_fn=data_fn, steps=1000)

def pred_fn():
    X, y = load_data("../data/ex3data1_X.txt", "../data/ex3data1_y.txt")

    out = {}
    for i in range(400):
        out["px"+str(i)] = [X[1800][i]]

    dataset = tf.data.Dataset.from_tensor_slices((out,))
    dataset = dataset.batch(1)
    return dataset.make_one_shot_iterator().get_next()

pre = estimator.predict(input_fn=pred_fn)
for p in pre:
    print(p["probabilities"])


