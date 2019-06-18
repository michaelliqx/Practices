import os
import numpy as np
import struct



class KNN:
    def __init__(self,X,Y):
        self.Xtr = X
        self.Ytr = Y

    def predict(self,test,k):
        num_test = test.shape[0]
        Ypred = np.zeros(num_test)


        for i in range(num_test):
            temp = np.zeros(len(np.unique(self.Ytr)))
            distance = np.sqrt(np.sum(np.square(self.Xtr - test[i,:]),axis=1))

            for j in range(k):
                min_dis = np.argmin(distance)
                temp[self.Ytr[min_dis]]+=1
                distance[min_dis] = 9999999999
            Ypred[i] = np.argmax(temp)
            # mindis = np.argmin(distance)
            # Ypred[i] = self.Ytr[mindis]

        ####################################################
        # no loop version                                  #
        # ab = np.dot(X, np.transpose(test))               #
        # a2 = np.sum(np.square(X),axis=1,keepdims=True)   #
        # b2 = np.sum(np.square(test),axis=1)              #
        # dists = np.sqrt(a2 + b2 - 2 * ab)                #
        # then find k nearest neighbor                     #
        ####################################################

        return Ypred


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


def main():
    # import data
    train, train_labels = load_mnist("/Users/michaelliqx/PycharmProjects/test/MNIST_data")
    test, test_labels = load_mnist("/Users/michaelliqx/PycharmProjects/test/MNIST_data", "t10k")
    # get the slice
    train_slice = 60000
    test_slice = 10000
    X = train[0:train_slice,:]
    Y = train_labels[0:train_slice]
    test = test[0:test_slice,:]
    test_labels = test_labels[0:test_slice]
    k=5
    knn = KNN(X,Y)
    res = knn.predict(test,k)
    acc = np.sum(np.equal(res,test_labels))/test.shape[0]
    print(acc)

if __name__ == '__main__':
    main()