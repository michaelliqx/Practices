from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def gen_data(N,D,K):
    X = np.zeros((N * K, D))  # data matrix (each row = single example)
    y = np.zeros(N * K, dtype='uint8')  # class labels
    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N)  # radius
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    # lets visualize the data:
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()
    return X,y

def calLoss(reg,W,scores,Y,num_samples):
    # use Softmax to classify

    exp_score = np.exp(scores)
    probs = exp_score/np.sum(exp_score,axis=1,keepdims=True)
    correct_logprob = -np.log(probs[range(num_samples),Y])
    data_loss = np.sum(correct_logprob)/num_samples
    reg_loss = 0.5 * reg * np.sum(W*W)
    loss = data_loss + reg_loss

    return loss,probs

def gradientDescent(probs,num_samples,X,Y,reg,W):
    dscores = probs
    dscores[range(num_samples),Y] -= 1
    dscores /= num_samples
    dW = np.dot(X.T,dscores)
    db = np.sum(dscores,axis=0,keepdims=True)
    dW += reg * W

    return dW,db,dscores

def softmax(X,Y,W,b,step_size,num_samples,reg,iter):
    for i in range(iter):
        scores = np.dot(X,W) + b

        # calculate loss
        loss,probs = calLoss(reg,W,scores,Y,num_samples)
        if (i%10==0):
            print("iteration %d: loss %f" % (i, loss))
        # gradient descent
        dW,db,dscores = gradientDescent(probs,num_samples,X,Y,reg,W)
        # update W,b
        W += -step_size * dW
        b += -step_size *db

    #evaluate
    scores = np.dot(X, W) + b
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == Y)))

    return dscores

def neuralNet(X,Y,D,K):
    # initialize parameters randomly
    h = 100  # size of hidden layer
    W = 0.01 * np.random.randn(D, h) # D: dimensionality   # X: (num_samples,D) W:(D,h)
    b = np.zeros((1, h))
    W2 = 0.01 * np.random.randn(h, K) # K: classes
    b2 = np.zeros((1, K))

    # some hyperparameters
    step_size = 1e-0
    reg = 1e-3  # regularization strength

    # gradient descent loop
    num_samples = X.shape[0]
    for i in range(10000):

        # evaluate class scores, [N x K]
        hidden_layer = np.maximum(0, np.dot(X, W) + b)  # note, ReLU activation hidden layer output
        scores = np.dot(hidden_layer, W2) + b2 # output

        # compute the class probabilities
        loss, probs = calLoss(reg, W, scores, Y, num_samples)
        if i % 1000 == 0:
            print("iteration %d: loss %f" % (i, loss))

        # compute the gradient on scores
        dscores = probs
        dscores[range(num_samples), Y] -= 1
        dscores /= num_samples

        # backpropate the gradient to the parameters
        # first backprop into parameters W2 and b2
        dW2 = np.dot(hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        # next backprop into hidden layer
        dhidden = np.dot(dscores, W2.T)
        # backprop the ReLU non-linearity
        dhidden[hidden_layer <= 0] = 0
        # finally into W,b
        dW = np.dot(X.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)

        # add regularization gradient contribution
        dW2 += reg * W2
        dW += reg * W

        # perform a parameter update
        W += -step_size * dW
        b += -step_size * db
        W2 += -step_size * dW2
        b2 += -step_size * db2

    # evaluate training set accuracy
    hidden_layer = np.maximum(0, np.dot(X, W) + b)
    scores = np.dot(hidden_layer, W2) + b2
    predicted_class = np.argmax(scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == Y)))

def main():

    #parameters
    N = 100  # number of points per class
    D = 2  # dimensionality
    K = 3  # number of classes
    reg = 1e-3 # lambda
    step_size = 1e-0
    iter = 200

    # generate dataset
    X,Y = gen_data(N,D,K)
    num_samples = X.shape[0]

    # initial parameters
    W = 0.01 * np.random.randn(D,K)
    b = np.zeros((1,K))

    # try softmax classifier
    dscores = softmax(X,Y,W,b,step_size,num_samples,reg,iter)

    # try neural network
    neuralNet(X,Y,D,K)
    

if __name__ == '__main__':
    main()