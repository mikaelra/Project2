import numpy as np

class NNlinreg:
    def __init__(
        self,
        X_data,
        Y_data,
        X_test,
        Y_test,
        n_hidden_neurons=50,
        n_categories=10,
        epochs=10,
        batch_size=100,
        eta=0.1,
        lmbd=0.0,

    ):
        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.X_test = X_test
        self.Y_test = Y_test

        self.n_inputs = X_data.shape[0]
        self.n_features = X_data.shape[1]
        self.n_hidden_neurons = n_hidden_neurons
        self.n_categories = n_categories

        self.epochs = epochs
        self.batch_size = batch_size
        self.iterations = self.n_inputs // self.batch_size
        self.eta = eta
        self.lmbd = lmbd
        self.r2scores = []

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.sigmoid_function(self.z_h)      # maybe change function

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        self.probabilities = self.linear(self.z_o)
        #print(self.probabilities)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias

        a_h = self.sigmoid_function(z_h)
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias

        probabilities = self.linear(z_o)
        return probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.sigmoid_function_d(self.a_h)
        # maybe change function

        self.output_weights_gradient = np.matmul(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.matmul(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient

        """
        print('error output:')
        print(error_output)
        print('output weights:')
        print(self.output_weights)
        print('hidden weights:')
        print(self.hidden_weights)
        """

    def predict(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities                 #np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        self.r2scores = []
        data_indices = np.arange(self.n_inputs)

        for i in range(self.epochs):
            for j in range(self.iterations):
                # pick datapoints with replacement
                chosen_datapoints = np.random.choice(
                    data_indices, size=self.batch_size, replace=False
                )

                # minibatch training data
                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.feed_forward()
                self.backpropagation()

            # Add the r2score for later plotting
            self.r2scores.append(self.r2score(self.X_test, self.Y_test))

            print('Epoch # %s done' %(i+1))
            print('R2: ', self.r2score(self.X_test, self.Y_test))
            print()

    def r2score(self, X_test, Y_test):
        y_bar = np.mean(Y_test)
        SSE = 0
        SSyy = 0
        for i in range(len(X_test)):
            SSE += (self.predict(X_test[i:i+1])[0][0] - Y_test[i:i+1][0][0])**2
        for i in range(len(X_test)):
            SSyy += (Y_test[i:i+1][0][0] - y_bar)**2

        return 1 - (SSE)/(SSyy)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_d(self, x):
        x[x<=0] = 0
        x[x>0] = 1
        return x

    def linear(self, x):
        return x

    def linear_d(self, x):
        return 1

    def sigmoid_function(self, x):
        return 1./(1 + np.exp(-x))

    def sigmoid_function_d(self, x):
        return self.sigmoid_function(x) * (1 - self.sigmoid_function(x))
