import time

class node:
    def __init__(self,data = None):
        self.data = data
        self.point = None

class linklist:
    def __init__(self):
        self.head = None


def quicksort(q):
    if len(q) <= 1:
        return q
    else:
        firstpart = [];
        secondpart = [];
        pivot = q[len(q)-1];
        i = 0
        for i in range(len(q)-1):
            if q[i]<=pivot:
                firstpart.append(q[i])
            else:
                secondpart.append(q[i])
        return quicksort(firstpart)+[pivot]+quicksort(secondpart)


def mergesort(q):

    mid = len(q)//2
    left = q[0:mid]
    right = q[mid:]
    return merge(mergesort(left),mergesort(right))


def merge(left,right):
    res = []
    i = 0
    j = 0
    while i < len(left) & j< len(right):
        if left[i] < right[j]:
            res.append(left[i])
            i+=1
        else:
            res.append(right[j])
            j+=1
    return res


def selectsort(q):
    for i in range(len(q)-1):
        for j in range(i,len(q)):
            if q[i]<q[j]:
                continue
            else:
                q[i],q[j] = q[j],q[i]
    return q

def bubblesort(q):
    for i in range(len(q)-1):
        for j in range(len(q)):
            if q[j]< q[j+1]:
                continue
            else:
                q[j],q[j+1] = q[j+1],q[j]

def lineartest():
    import tensorflow as tf
    import numpy as np

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    x = tf.placeholder(tf.float32,[None,2]) # dim1 could change (the number of samples to feed) but dim2 is clear, which is the number of features
    y = tf.placeholder(tf.float32,[None,1])

    w = tf.Variable(tf.random_normal([2,1]))
    b = tf.Variable(tf.random_normal([1]))


    out = tf.matmul(x,w) + b
    loss = tf.reduce_mean(tf.square(out - y))

    train = tf.train.AdagradOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            for j in range(4):
                sess.run(train,feed_dict={x: np.expand_dims(X[j],0) , y: np.expand_dims(Y[j],0)})
            loss_ = sess.run(loss, feed_dict={x:X,y:Y})
            print("step: %d, loss: %.3f" % (i, loss_))
        print("X: %r" % X)
        print("pred: %r" % sess.run(out, feed_dict={x: X}))

def hiddenlayertest():
    import tensorflow as tf
    import numpy as np

    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    Y = np.array([[0],[1],[1],[0]])

    x = tf.placeholder(tf.float32,[None,2])
    y = tf.placeholder(tf.float32,[None,1])

    w1_1 = tf.Variable(tf.random_normal([2,1]))
    w1_2 = tf.Variable(tf.random_normal([2,1]))
    w2 = tf.Variable(tf.random_normal([2,1]))

    b1_1 = tf.constant(0.1,shape=[1])
    b1_2 = tf.constant(0.1,shape=[1])
    b2 = tf.constant(0.1,shape=[1])

    h1 = tf.nn.relu(tf.matmul(x,w1_1)+b1_1)
    h2 = tf.nn.relu(tf.matmul(x,w1_2)+b1_2)

    hidden = tf.concat([h1,h2],1) # connect 2 vector
    out = tf.matmul(hidden,w2) + b2

    loss = tf.reduce_mean(tf.square(out- y))

    train = tf.train.AdagradOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            for j in range(4):
                sess.run(train, feed_dict={x:np.expand_dims(X[j],0), y: np.expand_dims(Y[j],0)})
            loss_ = sess.run(loss, feed_dict={x:X,y:Y})
            print("step: %d, loss: %.3f" % (i, loss_))
        print("X: %r" % X)
        print("pred: %r" % sess.run(out, feed_dict={x: X}))

def main():
    # #linklist
    # list = linklist()
    # list.head = node(1)
    # node2 = node(2)
    # list.head.point = node2
    # print(list.head.point.data)
    #
    # q = [3,5,4,6,9,7,8,0,1,2,3]
    # a = quicksort(q)
    # starttime = time.clock()
    # print("quicksort:",a)
    # endtime = time.clock()
    # print("running time: {} s".format(endtime - starttime))
    # b = selectsort(q)
    # starttime = time.clock()
    # print("selectsbort:", b)
    # endtime = time.clock()
    # print("running time: {} s".format(endtime - starttime))
    # c = selectsort(q)
    # starttime = time.clock()
    # print("bubblesbort:", c)
    # endtime = time.clock()
    # print("running time: {} s".format(endtime - starttime))
    # d = selectsort(q)
    # starttime = time.clock()
    # print("mergesbort:", d)
    # endtime = time.clock()
    # print("running time: {} s".format(endtime - starttime))

    # tensorflow test
    # lineartest()
    # hiddenlayertest()

    pass



if __name__ == "__main__":
    main()