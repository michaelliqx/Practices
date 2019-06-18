import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import os
import struct

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte'
                               % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',
                                 lbpath.read(8))
        labels = np.fromfile(lbpath,
                             dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',
                                               imgpath.read(16))
        images = np.fromfile(imgpath,
                             dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

def get_Batch(data,label,batch_size):
    input_queue = tf.train.slice_input_producer([data,label],num_epochs=100,shuffle=True,seed=1,capacity=32)
    x_batch, y_batch = tf.train.batch(input_queue,batch_size = batch_size, num_threads=1,allow_smaller_final_batch=False)
    return x_batch,y_batch



def buildmodel(train,train_labels):
    # input and output
    X = tf.placeholder(dtype=tf.float32,shape = [None,784],name = "X")
    Y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name="Y")

    # build model
    W1 = tf.get_variable("W1",[784,128], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1",[128],initializer=tf.zeros_initializer())
    A1 = tf.nn.relu(tf.matmul(X, W1) + b1, name="A1")
    W2 = tf.get_variable("W2",[128,64],initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2",[64],initializer=tf.zeros_initializer())
    A2 = tf.nn.relu(tf.matmul(A1, W2) + b2, name="A2")
    W3 = tf.get_variable("W3",[64,10],initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3",[10],initializer=tf.zeros_initializer())
    Z3 = tf.matmul(A2,W3)+b3

    # define the cost
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Z3,labels=Y))
    trainer = tf.train.AdagradOptimizer(learning_rate=1e-3).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Z3, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='evaluation')

    x_batch, y_batch = get_Batch(train, train_labels, 64)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # data, label = sess.run([x_batch, y_batch])
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)
        epoch = 0


        while not coord.should_stop():
            data, label = sess.run([x_batch, y_batch])
            labels = np.zeros([64,10])
            for j in range(len(label)):
                labels[j,label[j]] = 1
            sess.run(trainer,feed_dict={X:data,Y:labels})
            train_accuracy = accuracy.eval({X: data, Y: labels})
            # test_accuracy = accuracy.eval({X: test_x, Y: test_y})
            print("Epoch %d, Training accuracy %g" % (epoch, train_accuracy))

            epoch += 1



def main():
    # load MNIST data set
    train, train_labels = load_mnist("/Users/michaelliqx/PycharmProjects/test/MNIST_data")
    test, test_labels = load_mnist("/Users/michaelliqx/PycharmProjects/test/MNIST_data","t10k")
    buildmodel(train,train_labels)



if __name__ == '__main__':
    main()
