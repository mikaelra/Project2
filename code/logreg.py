from realdatagen import getrealdata
from LogisticRegGD import LogisticRegGD
import matplotlib.pyplot as plt
import numpy as np


def showimage(X, y):
    img = np.reshape(X, (40, 40))
    if y[0] == 0:
        s = 'disordered'
    elif y[0] == 1:
        s = 'ordered'
    else:
        s = 'err0r, check algorithm'

    plt.title(s)
    plt.imshow(img, cmap ='hot', vmin= -1, vmax = 1)
    plt.colorbar()
    plt.show()

if __name__ == '__main__':

    # Uncomment parts of the code to run it
    # The ones with #PLOT takes a very long time without the shortening of the testsamples

    X_train, X_test, Y_train, Y_test = getrealdata(train_to_test_ratio=0.9)

    """
    # This takes 1000 random samples so testing takes less time
    c = np.random.choice(len(X_test), 1000)
    Y_test = Y_test[c]
    X_test = X_test[c]
    """

    print('Lengde på X_train:' + str(len(X_train)))
    print('Lengde på X_test:' + str(len(X_test)))
    print()

    """
    # Shows some images of the IsingData
    # Says if it is ordered or not

    for i in range(100, 110):
        showimage(X_train[i], Y_train[i])
    """

    from sklearn.linear_model import LogisticRegression

    """
    # PLOT
    logreg = LogisticRegression(random_state=0, solver='saga', max_iter=100)
    logreg.fit(X_train, Y_train)
    print('Accuracy score of sklearn LogReg using Stochastic Average Descent and max_iter of 100:')
    print(logreg.score(X_test, Y_test))
    # Outputs 0.7053117408906883 when training to test ratio is 0.95
    """

    # Under is a lot of plots of different parameters to see what scores they get

    """
    for eta in [0.001, 1e-5, 1e-8, 1e-10, 1e-14]:
        # PLOT
        # Using LogReg with Stochastic Gradient descent
        lr = LogisticRegGD(X_train, Y_train, X_test, Y_test,eta=eta ,plotacc=True, max_iter=10)
        lr.fit()
        plt.plot(lr.accuracies)
        plt.title('Accuracy over 1000 test data, 10 iterations, %s eta' %eta)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()
    """

    """
    for iter in [10, 100, 250, 500, 1000]:
        # PLOT
        # Using LogReg with Stochastic Gradient descent
        lr = LogisticRegGD(X_train, Y_train, X_test, Y_test, eta=1e-8,plotacc=True, max_iter=iter)
        lr.fit()
        plt.plot(lr.accuracies)
        plt.title('Accuracy over 1000 test data, %d iterations' %iter)
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()
    """


    """
    for i in range(5):
        lr = LogisticRegGD(X_train, Y_train, X_test, Y_test, max_iter=250)
        lr.fit()
        print()
        print('Accuracy score of test data: ' + str(lr.score(X_test, Y_test)))
        print('------------------------')

        lr = LogisticRegGD(X_train, Y_train, X_test, Y_test, max_iter=500)
        lr.fit()
        print()
        print('Accuracy score of test data: ' + str(lr.score(X_test, Y_test)))
        print('------------------------')

        lr = LogisticRegGD(X_train, Y_train, X_test, Y_test, max_iter=1000)
        lr.fit()
        print()
        print('Accuracy score of test data: ' + str(lr.score(X_test, Y_test)))
        print('------------------------')
    #print('Accuracy score of training data: ' + str(lr.score(X_train, Y_train)))
    """

    """
    print()
    print()

    lr = LogisticRegGD(X_train, Y_train, X_test, Y_test, max_iter=1000)
    lr.fit()
    print()
    print('Accuracy score of test data: ' + str(lr.score(X_test, Y_test)))
    print('Accuracy score of training data: ' + str(lr.score(X_train, Y_train)))
    """
