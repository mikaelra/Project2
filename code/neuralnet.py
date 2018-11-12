from NN import NN
from realdatagen import *
import numpy as np
import matplotlib.pyplot as plt
from logreg import showimage

# Generate test the real data
X_train, X_test, Y_train, Y_test = getrealdata(train_to_test_ratio=0.5)

"""
"""
# Set some variables to use in the neural network
epochs = 10
eta = 0.1
batch_size = 100
n_hidden_neurons = 50
n_categories = 1
lmbd = 0


# Sets up the network
dnn = NN(X_train, Y_train, X_test, Y_test, eta=eta, lmbd=lmbd, epochs=epochs, batch_size=batch_size,
n_hidden_neurons=n_hidden_neurons, n_categories=n_categories)

# Train the network
dnn.train()

# Record the data which gets predicted wrong
wronglist = []
for i in range(len(X_test)):
    y_p = round(dnn.feed_forward_out(X_test[i:i+1])[0][0])
    if y_p != Y_test[i:i+1][0][0]:
        wronglist.append(i)
        #showimage(X_test[i:i+1], Y_test[i:i+1])

# This shows the progress of the accuracy over the epochs
plt.plot(dnn.accuracyscores)
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

# Print the score of the network when used on test data
print('Accuracy of my neural net:' + str(dnn.class_score(X_test, Y_test)))

# Doing the same thing with keras
"""
"""
import tensorflow as tf
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(50, activation='sigmoid', input_dim=X_train.shape[1]))
model.add(tf.keras.layers.Dropout(0.0))
model.add(tf.keras.layers.Dense(Y_train.shape[1], activation='sigmoid'))


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(
    X_train,
    Y_train,
    epochs=10,
    batch_size=16,
    validation_data=[X_test, Y_test],
    #callbacks=[tensorboard_callback],
)
wronglist2 = []
for i in range(len(X_test)):
    y_p = round(float(model.predict(X_test[i:i+1])))
    if y_p != Y_test[i:i+1][0][0]:
        wronglist2.append(i)
        showimage(X_test[i:i+1], Y_test[i:i+1])



print('accuracy:', model.evaluate(X_test, Y_test))
# Gets accuarcy of approx 0.9995

print('wrong prediction in my NN:')
print(wronglist)
print('wrong in TF NN:')
print(wronglist2)
