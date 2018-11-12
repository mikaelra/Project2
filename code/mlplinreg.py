from NNlinreg import NNlinreg
from datagen import genIsingData
import numpy as np
import matplotlib.pyplot as plt

L=40; amount=100000; testamount=0.9
X_train, X_test, Y_train, Y_test = genIsingData(L=L, amount=amount, testamount=testamount)

# Set some variables to use in the neural network
epochs = 100
eta = 0.001
batch_size = 100
n_hidden_neurons = 150
n_categories = 1
lmbd = 0


"""
"""
# Sets up the network
dnn = NNlinreg(X_train, Y_train, X_test, Y_test,eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)

# Train the network
dnn.train()

# Samples
c = np.random.choice(len(X_test), 5)

# This tests predicts the Ising energies from the test data and compares it with the real value
for i in c:
    print('Sample # ' + str(i))
    print('Predicting ising energy:')
    print(dnn.predict(X_test[i:i+1]))
    print('Actual:')
    print(Y_test[i:i+1])
    print()
"""
"""

print('R2-score of my NN:')
print(dnn.r2score(X_test, Y_test))

plt.plot(dnn.r2scores)
plt.xlabel('Generation')
plt.ylabel('R2-score')
plt.show()



import tensorflow as tf

# Doing the same thing with keras

import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(150, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='linear'))


model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
model.fit(
    X_train,
    Y_train,
    epochs=100,
    batch_size=100,
    validation_data=[X_test, Y_test],
    #callbacks=[tensorboard_callback],
)

print()
for i in c:
    print('Sample # ' + str(i))
    print('Predicting ising energy:')
    print(model.predict(X_test[i:i+1]))
    print('Actual energy:')
    print(Y_test[i:i+1])
    print()


def r2score(model, X_test, Y_test):
    y_bar = np.mean(Y_test)
    SSE = 0
    SSyy = 0
    for i in range(len(X_test)):
        SSE += (model.predict(X_test[i:i+1])[0][0] - Y_test[i:i+1][0][0])**2
    for i in range(len(X_test)):
        SSyy += (Y_test[i:i+1][0][0] - y_bar)**2

    return 1 - (SSE)/(SSyy)

print('R2score of TF NN:')
print(r2score(model, X_test, Y_test))

"""
"""
