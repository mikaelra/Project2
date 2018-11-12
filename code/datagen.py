import numpy as np
import scipy.sparse as sp
np.random.seed(12)

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')


def genIsingData(L=40, amount=10000, testamount=0.95):
    ### define Ising model aprams
    # system size
    L=40

    # create 10000 random Ising states
    amount=10000
    states=np.random.choice([-1, 1], size=(amount,L))

    def ising_energies(states,L):
        """
        This function calculates the energies of the states in the nn Ising Hamiltonian
        """
        J=np.zeros((L,L),)
        for i in range(L):
            J[i,(i+1)%L]-=1.0
        # compute energies
        E = np.einsum('...i,ij,...j->...',states,J,states)

        return E
    # calculate Ising energies
    energies=ising_energies(states,L)

    def isingmatrix(onestate):
        """
        This function computes the Xi matrix, so that we can apply linear regression to it
        """
        if len(onestate.shape) == 1:
            N = onestate.shape[0]
        else:
            N = onestate.shape[1]
        X = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                X[j][k] = onestate[j]*onestate[k]

        # We flatten the matrix to make it a 1D vector input
        return X.flatten()


    XIstates = np.zeros((amount,L*L))
    for i in range(amount):
        XIstates[i] = isingmatrix(states[i])


    # Splitting up in training and test data
    Xtrain, Xtest, Ytrain, Ytest = (XIstates[0:int(testamount*amount)], XIstates[int(testamount*amount)::],
                                    energies[0:int(testamount*amount)], energies[int(testamount*amount)::])

    Ytrain = np.expand_dims(Ytrain, axis=1)
    Ytest = np.expand_dims(Ytest, axis=1)

    return Xtrain, Xtest, Ytrain, Ytest

if __name__ == '__main__':

    # The commented out sections provide some images and data used in the project
    # This one provides the bootstrap-data for task b)


    L=40; amount=10000; testamount=0.95

    Xtrain, Xtest, Ytrain, Ytest = genIsingData(L=L, amount= amount, testamount=testamount)

    # This includes p1 as directory, so we can use the modules from project 1
    import os
    s = os.path.dirname(os.path.abspath(__file__))
    s+= '/p1code'
    import sys
    sys.path.append(s)
    import matplotlib.pyplot as plt


    from OLSLinearModel import *
    from LassoLinearModel import *


    # Adapted the OLSmodel a little bit to be more general
    # Did not do this for all methods, so I will use sklearn
    # for most of this
    """
    OLSmodel = OLSLinearModel()
    OLSmodel.fit(Xtrain, Ytrain)
    betaimg = np.reshape(OLSmodel.beta, (L, L))
    plt.title('OLSModel of coefficients')
    plt.imshow(betaimg, cmap ='hot')
    plt.colorbar()
    plt.show()
    """
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.metrics import mean_squared_error

    """
    for i in [0.9, 0.1 , 0.001, 0.00001]:
        clfl = Lasso(alpha=i)
        clfl.fit(Xtrain, Ytrain)
        print('R2 score of Lasso with alpha = ' + str(i))
        print(clfl.score(Xtest, Ytest))
        print('MSE score of Lasso with alpha = ' + str(i))
        print(mean_squared_error(clfl.predict(Xtest), Ytest))
        betalimg = np.reshape(clfl.coef_, (L, L))
        plt.title('LassoModel of coefficients, alpha = ' + str(i))
        plt.imshow(betalimg, cmap ='hot')
        plt.colorbar()
        plt.show()
    """


    """
        clfr = Ridge(alpha=i)
        clfr.fit(Xtrain, Ytrain)
        print('R2 score of Ridge with alpha = ' + str(i))
        print(clfr.score(Xtest, Ytest))
        print('MSE score of Ridge with alpha = ' + str(i))
        print(mean_squared_error(clfr.predict(Xtest), Ytest))
        betarimg = np.reshape(clfr.coef_, (L, L))
        plt.title('RidgeModel of coefficients, alpha = ' + str(i))
        plt.imshow(betarimg, cmap ='hot')
        plt.colorbar()
        plt.show()
    """

    # Resampling using cross-validation

    # Put all the data in to Xtrain and Ytrain
    Xtrain, Xtest, Ytrain, Ytest = genIsingData(L=L, amount=amount, testamount=1)

    # Define a simple bootstrapping algorithm
    def bootstrap(Xtrain, Ytrain, model, alpha, b):

        for i in range(b):
            c = np.random.choice(len(Xtrain))

            bs_X = Xtrain[c:c+1]
            bs_y = Ytrain[c:c+1]

            clf = model(alpha=alpha)
            clf.fit(bs_X, bs_y)
            y_tilde = clf.predict(bs_X)

            if len(y_tilde.shape) == 1:
                y_tilde = np.expand_dims(y_tilde, axis=1)


            if i == 0:
                y_tilde_matrix = y_tilde
            else:
                y_tilde_matrix = np.concatenate([y_tilde_matrix, y_tilde], axis=1)

        #compute expected value in each x over the bootstrapsamples
        E_L = (np.mean(y_tilde_matrix, axis=1, keepdims=True))

        # compute bias
        bias = np.mean((Ytrain - E_L)**2)

        # compute variance
        var = np.mean(np.mean((y_tilde_matrix - E_L)**2, axis=1, keepdims=True))

        print('alpha: %f, bootstraps: %i' %(alpha, i+1))
        print("VAR: %f" % var)
        print("BIAS: %f" % bias)


    # Produce the data
    b = 500

    print('Ridge:')
    bootstrap(Xtrain, Ytrain, Ridge, 1, b)
    print()
    bootstrap(Xtrain, Ytrain, Ridge, 0.1, b)
    print()
    bootstrap(Xtrain, Ytrain, Ridge, 0.001, b)
    print()
    bootstrap(Xtrain, Ytrain, Ridge, 1e-5, b)
    print()
    print()
    print('Lasso:')
    bootstrap(Xtrain, Ytrain, Lasso, 1, b)
    print()
    bootstrap(Xtrain, Ytrain, Lasso, 0.1, b)
    print()
    bootstrap(Xtrain, Ytrain, Lasso, 0.001, b)
    print()
    bootstrap(Xtrain, Ytrain, Lasso, 1e-5, b)
