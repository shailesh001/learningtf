import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

DATA_DIR = '../../data'

data = input_data.read_data_sets(DATA_DIR, one_hot=False)
x_data, y_data = data.train.images, data.train.labels.astype(np.int32)
x_test, y_test = data.test.images, data.test.labels.astype(np.int32)

NUM_STEPS = 2000
MINIBATCH_SIZE = 128

feature_columns = learn.infer_real_valued_columns_from_input(x_data)

dnn = learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[200],
    n_classes=10,
    optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.2)
)

dnn.fit(x=x_data, y=y_data, steps=NUM_STEPS, batch_size=MINIBATCH_SIZE)

test_acc = dnn.evaluate(x=x_test, y=y_test, steps=1)["accuracy"]
print('test accuracy: {}'.format(test_acc))
