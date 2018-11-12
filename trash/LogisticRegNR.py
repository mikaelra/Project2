import numpy as np



class LogisticRegNR:
    # All variables

    def __init__(
        self,
        X_train,
        Y_train,
        X_test,
        Y_test,
        iterations = 10

    ):

        nbetas = None
        beta = None
        self.X = X_train
        self.y = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.iterations = iterations
        p = None
        W = None

    def fit(self):
        N = self.X.shape[0]
        self.X = np.c_[np.ones(N), self.X]
        # X is now the shape N x (p + 1)

        # Calculate amount of betas
        self.nbetas = self.X.shape[1]
        self.beta = np.zeros((self.nbetas, 1))


        # Need to split up X into smaller partitions

        for k in range(self.iterations):
            # Calculate p-s.
            self.p = np.zeros(N)
            for i in range(N):
                sum = self.beta[0]
                for j in range(self.nbetas - 1):
                    sum += self.beta[j+1] * self.X[i][j]
                self.p[i] = (np.exp(sum))/(1 + np.exp(sum))

            self.p = np.expand_dims(self.p, axis=1)
            #Calculate W
            self.W = np.eye(N)
            self.W *= self.p.T *(1-self.p.T)

            # Første metode
            eta = 10e-5

            part1 = np.linalg.pinv(np.dot(np.dot(self.X.T, self.W), self.X))
            part100 = np.dot(self.X.T, (self.y -self.p))
            self.beta = np.add(self.beta, eta*np.dot(part1, part100)) #np.dot(part1, part100)
            """

            # Calculate z
            z = np.add(np.dot(self.X, self.beta) , np.dot(np.linalg.pinv(self.W), np.subtract(self.y, self.p)))
            # Formula 4.26 in Hastie et al
            # Calculate new beta

            part1 = np.linalg.pinv(np.dot(np.dot(self.X.T, self.W), self.X))
            part2 = np.dot(self.X.T, self.W)
            part12 = np.dot(part1, part2)

            self.beta = np.dot(part12, z)
            """
            """

            """
            print('Score etter %s iterasjoner' %(k+1))
            print(self.score(self.X_test, self.Y_test))
            print('Største beta:')
            print(np.amax(self.beta))
            print('Minste beta:')
            print(np.amin(self.beta))
            print()

    def predict(self, X):
        sum = self.beta[0]

        for j in range(self.nbetas -1):
            sum += self.beta[j+1] * X[j]
        return (np.exp(sum) )/( 1 + np.exp(sum))

    def score(self, X, y):
        hit = 0.
        bom = 0.
        for i in range(len(y)):
            if (round(self.predict(X[i])[0]) == y[i][0]):
                #print(self.predict(X[i])[0])
                hit+=1
            elif (round(self.predict(X[i])[0]) == np.abs(y[i][0] - 1)):
                bom +=1
            else:
                print(self.predict(X[i])[0])
        return hit/len(y), bom/len(y)
