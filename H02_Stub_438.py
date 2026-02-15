# Govan Henry CMSC 438 HW2  2/12/2026

#!/usr/bin/env python
# coding: utf-8
# Python library imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd;

# Function definitions for reading data and training the model
def read_csv_convert_to_numpy(fileName='carSUV_normalized.csv'):
    # input: path to file name
    # outputs:
    # numpy_x, a 2D numpy array with two columns (first column: ZeroToSixty
    # feature, second column: PowerHP feature), one sample per row
    # numpy_y, a 2D numpy array with one column, containint +1 if a car, -1 if an
    # SUV, one sample per row

    # YOUR CODE HERE
    df = pd.read_csv(fileName, index_col='SampleName')
    numpy_x = df[['ZeroToSixty', 'PowerHP']].to_numpy()
    y_raw = df['IsCar'].to_numpy()
    numpy_y = np.where(y_raw == 1, 1, -1).reshape(-1, 1)

    return numpy_x, numpy_y


def calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y):
    # inputs:
    # w: a numpy 2D array (#features-by-1)
    # numpy_x: a numpy 2D numpy array (#samples-by-#features)
    # numpy_y: a numpy 2D numpy array (#samples-by-1)
    # output:
    # single real number in range [0.0,1.0], the number of errors dividied by
    # samples

    # YOUR CODE HERE
    n = numpy_x.shape[0]
    errors = 0

    for i in range(n):
        x = numpy_x[i].reshape(2, 1)
        y_true = numpy_y[i][0]

        pred_raw = float(w.T @ x)
        y_pred = 1 if pred_raw > 0 else -1

        if y_pred != y_true:
            errors += 1

    error_rate = errors / n
    return error_rate


def train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01):
    #inputs: numpy_x, numpy_y - features, classes
    #output: a 2D numpy array of size (#features, 1), containing final weights,
    #after training is complete

    #for the input from 'carSUV_normalized.csv' processed by
    #read_csv_convert_to_numpy, these two prints should return (10,2) and (10,1)
    print(numpy_x.shape) # (#samples=10,#features=2)
    print(numpy_y.shape) # (#samples=10,1) (+1 for car, -1 for SUV)

    # YOUR CODE HERE
    w = np.random.randn(2,1)
    n = numpy_x.shape[0]

    for epoch in range(n_epochs):
        for i in range(n):
            x = numpy_x[i].reshape(2, 1)
            y_true = numpy_y[i][0]

            pred_raw = float(w.T @ x)
            y_pred = 1 if pred_raw > 0 else -1

            if y_pred != y_true:
                w = w + c * (y_true - y_pred) * x

        err = calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)
        print("Epoch", epoch, "error rate:", err)

    #plot_trained_w_and_dataset(numpy_x, numpy_y, w)
    return w;


if __name__ == "__main__":
    # Below are some helper functions and code that may be useful to visualize your
    # progress
    def plot_trained_w_and_dataset(numpy_x, numpy_y, w):
        samples_class1 = numpy_y.flatten()==1
        samples_class0 = numpy_y.flatten()==-1
        plt.scatter(numpy_x[samples_class1,0], numpy_x[samples_class1,1], c='red')
        plt.scatter(numpy_x[samples_class0,0], numpy_x[samples_class0,1],
        c='green')
        plt.xlabel('ZeroTwoSixty')
        plt.ylabel('PowerHP')
        if (w[1]==0): #weights are (something,0); feature x2 doesn't matter
            x2_line = np.linspace(-2, 2, 100)
            x1_line = 0*x2_line;
        else:
            x1_line = np.linspace(-2, 2, 100)
            x2_line = (-w[0] * x1_line) / w[1]
        # Create a blue line based on the equation
        plt.plot(x1_line, x2_line, c='blue')
        plt.show()

    # Testing read_csv_convert_to_numpy & calc_error_rate_for_single_vector_w
    # see HW2 slides for expected output
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    np.random.seed(3) # to fix randomness
    random_w = np.random.randn(2,1)
    print("Random weights array shape",random_w.shape)
    print("Random weights values\n",random_w)
    error_rate_random_weights = calc_error_rate_for_single_vector_w(random_w,
    numpy_x, numpy_y)
    print("Error rate for random weights",error_rate_random_weights)

    # Testing train_and_evaluate; Running data reading, model training, and
    # plotting the linear model over the dataset, using the functions defined above.
    # see HW2 slides for expected output
    np.random.seed(8) # to eliminate randomness
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    trained_w = train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01);
    print(trained_w)
    plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w);
