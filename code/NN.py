import numpy as np

class NN:
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

        self.accuracyscores = []

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.n_features, self.n_hidden_neurons)
        self.hidden_bias = np.zeros(self.n_hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.n_hidden_neurons, self.n_categories)
        self.output_bias = np.zeros(self.n_categories) + 0.01

    def feed_forward(self):
        # feed-forward for training
        self.z_h = np.matmul(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.sigmoid_function(self.z_h)

        self.z_o = np.matmul(self.a_h, self.output_weights) + self.output_bias

        #exp_term = np.exp(self.z_o)
        self.probabilities = self.sigmoid_function(self.z_o)                 #exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def feed_forward_out(self, X):
        # feed-forward for output
        z_h = np.matmul(X, self.hidden_weights) + self.hidden_bias

        a_h = self.sigmoid_function(z_h)
        z_o = np.matmul(a_h, self.output_weights) + self.output_bias


        #exp_term = np.exp(z_o)
        #probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return self.sigmoid_function(z_o)             #probabilities

    def backpropagation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.matmul(error_output, self.output_weights.T) * self.sigmoid_function_d(self.z_h)

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
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.feed_forward_out(X)
        return probabilities

    def train(self):
        self.accuracyscores = []
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

            print()
            self.accuracyscores.append(self.class_score(self.X_test, self.Y_test))
            print('Epoch # %s done' %(i+1))
            print('Accuracy: ' + str(self.class_score(self.X_test, self.Y_test)))

    def class_score(self, X_test, Y_test):
        hit = 0
        for i in range(len(X_test)):
            guess = round(self.feed_forward_out(X_test[i:i+1])[0][0])
            target = Y_test[i:i+1][0][0]
            if guess - target < 1e-14:
                hit +=1
        return hit/len(X_test)


    def sigmoid_function(self, x):
        return 1./(1 + np.exp(-x))

    def sigmoid_function_d(self, x):
        return self.sigmoid_function(x) * (1 - self.sigmoid_function(x))
