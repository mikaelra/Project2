import numpy as np
import sys


class LogisticRegGD:
    # All variables

    def __init__(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        max_iter = 100,
        eta = 0.00001,
        eps = 1,
        plotacc = False

    ):

        nbetas = None
        beta = None
        self.X = X_train
        self.y = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.max_iter = max_iter
        self.eta = eta
        self.eps = eps
        self.accuracies = []
        self.plotacc = plotacc

    def fit(self):
        self.accuracies = []
        print('Started fitting...')
        N = self.X.shape[0]
        self.X = np.c_[np.ones(N), self.X]
        # X is now the shape N x (p + 1)

        # Calculate amount of betas
        self.nbetas = self.X.shape[1]
        self.beta = np.zeros((self.nbetas, 1))

        for k in range(self.max_iter):
            sys.stdout.write('\rIteration %d / %d' %((k+1), self.max_iter))
            sys.stdout.flush()

            if self.X.shape[0] > 100:
                c = np.random.choice(self.X.shape[0], 100)
                X = self.X[c]
                y = self.y[c]
            else:
                X = self.X
                y = self.y

            #print(np.dot(X, self.beta).shape)
            #print(self.y.shape)
            delta_C = 2 * np.dot(X.T, (np.dot(X, self.beta) - y))
            self.beta = self.beta - self.eta * delta_C

            # Calculating accuracies
            if self.plotacc:
                self.accuracies.append(self.score(self.X_test, self.Y_test))

            if np.linalg.norm(delta_C) <= self.eps:
                print('Done iterating after %d iterations' %(k+1))
                break

    def probability(self, X):
        sum = self.beta[0]
        for j in range(self.nbetas -1):
            sum += self.beta[j+1] * X[j]

        #print(1./(1 + np.exp(-sum)))
        return 1./(1 + np.exp(-sum))
        #return (np.exp(sum) )/( 1 + np.exp(sum))


    def predict(self, X):
        return np.around(self.probability(X))

    def score(self, X, y, iterprint=False):
        print()
        print('Started computing accuracy score ...')
        hit = 0.
        bom = 0.
        for i in range(len(y)):
            if iterprint:
                sys.stdout.write('\rCalculating %d / %d' %((i+1), len(y)))
                sys.stdout.flush()

            if (self.predict(X[i])[0] - y[i][0] <= 1e-10):
                #print(self.predict(X[i])[0])
                hit+=1
            elif (self.predict(X[i])[0] - np.abs(y[i][0] - 1) <= 1e-10):
                bom +=1
            else:
                print('Not 1 or 0')
                print(self.predict(X[i])[0])
        return hit/len(y)
