import numpy as np

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')

#np.random.seed(2) # shuffle random seed generator

# Ising model parameters
L=40 # linear system size
J=-1.0 # Ising interaction
T=np.linspace(0.25,4.0,16) # set of temperatures
T_c=2.26 # Onsager critical temperature in the TD limit

##### prepare training and test data sets

import pickle,os
from sklearn.model_selection import train_test_split


def getrealdata(train_to_test_ratio=0.5):
    # Returns X_train, X_test, Y_train, Y_test

    # training samples
    ###### define ML parameters
    num_classes=2

    # load data
    file_name = "/Ising2DFM_reSample_L40_T=All.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    data = pickle.load(open('IsingData'+file_name,'rb')) # pickle reads the file and returns the Python object (1D array, compressed bits)
    data = np.unpackbits(data).reshape(-1, 1600) # Decompress array and reshape for convenience
    data=data.astype('int')
    data[np.where(data==0)]=-1 # map 0 state to -1 (Ising variable can take values +/-1)

    file_name = "/Ising2DFM_reSample_L40_T=All_labels.pkl" # this file contains 16*10000 samples taken in T=np.arange(0.25,4.0001,0.25)
    labels = pickle.load(open('IsingData'+file_name,'rb')) # pickle reads the file and returns the Python object (here just a 1D array with the binary labels)

    # divide data into ordered, critical and disordered
    X_ordered=data[:70000,:]
    Y_ordered=labels[:70000]

    X_critical=data[70000:100000,:]
    Y_critical=labels[70000:100000]

    X_disordered=data[100000:,:]
    Y_disordered=labels[100000:]

    del data,labels

    # define training and test data sets
    X=np.concatenate((X_ordered,X_disordered))
    Y=np.concatenate((Y_ordered,Y_disordered))

    # pick random data points from ordered and disordered states
    # to create the training and test sets
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=train_to_test_ratio)

    # Add dimension to the set
    Y_train = np.expand_dims(Y_train, axis=1)
    Y_test = np.expand_dims(Y_test, axis=1)
    # full data set
    #X=np.concatenate((X_critical,X))
    #Y=np.concatenate((Y_critical,Y))

    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    X_train, X_test, Y_train, Y_test = getrealdata()
    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)
    print()
    print(X_train.shape[0], 'train samples')
    #print(X_critical.shape[0], 'critical samples')
    print(X_test.shape[0], 'test samples')
