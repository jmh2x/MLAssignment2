#!/usr/bin/env python
# coding: utf-8

# Python library imports
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
import pandas as pd;

# -------------------------------------------------------------------
# Function definitions for reading data and training the model
# -------------------------------------------------------------------

def read_csv_convert_to_numpy(fileName='carSUV_normalized.csv'):
    # input: path to file name
    # outputs:
    # numpy_x: 2D numpy array (10×2) with features
    # numpy_y: 2D numpy array (10×1) with +1 for car, -1 for SUV

    df = pd.read_csv(fileName, index_col='SampleName')

    numpy_x = df[['ZeroToSixty', 'PowerHP']].to_numpy()

    y_raw = df['IsCar'].to_numpy()
    numpy_y = np.where(y_raw == 1, 1, -1).reshape(-1, 1)

    return numpy_x, numpy_y


def calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y):
    # inputs:
    # w: numpy 2D array (#features-by-1)
    # numpy_x: numpy 2D array (#samples-by-#features)
    # numpy_y: numpy 2D array (#samples-by-1)
    # output: real number in [0,1], error rate

    n = numpy_x.shape[0]
    errors = 0

    for i in range(n):
        x = numpy_x[i].reshape(2, 1)
        y_true = numpy_y[i][0]

        pred_raw = float(w.T @ x)
        y_pred = 1 if pred_raw > 0 else -1

        if y_pred != y_true:
            errors += 1

    return errors / n


def train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01):
    # inputs: numpy_x, numpy_y
    # output: final trained weight vector (2×1)

    print(numpy_x.shape)  # (#samples=10,#features=2)
    print(numpy_y.shape)  # (#samples=10,1)

    # initialize weights randomly
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

        # print error after each epoch
        err = calc_error_rate_for_single_vector_w(w, numpy_x, numpy_y)
        print("Epoch", epoch, "error rate:", err)

    return w;


# -------------------------------------------------------------------
# Helper visualization functions (unchanged)
# -------------------------------------------------------------------

if __name__ == "__main__":

    def plot_trained_w_and_dataset(numpy_x, numpy_y, w):
        samples_class1 = numpy_y.flatten()==1
        samples_class0 = numpy_y.flatten()==-1
        plt.scatter(numpy_x[samples_class1,0], numpy_x[samples_class1,1], c='red')
        plt.scatter(numpy_x[samples_class0,0], numpy_x[samples_class0,1], c='green')
        plt.xlabel('ZeroTwoSixty')
        plt.ylabel('PowerHP')

        if (w[1]==0): #weights are (something,0); feature x2 doesn't matter
            x2_line = np.linspace(-2, 2, 100)
            x1_line = 0*x2_line;
        else:
            x1_line = np.linspace(-2, 2, 100)
            x2_line = (-w[0] * x1_line) / w[1]

        plt.plot(x1_line, x2_line, c='blue')
        plt.show()

    # Testing read_csv_convert_to_numpy & calc_error_rate_for_single_vector_w
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    np.random.seed(3) # to fix randomness
    random_w = np.random.randn(2,1)
    print("Random weights array shape",random_w.shape)
    print("Random weights values\n",random_w)
    error_rate_random_weights = calc_error_rate_for_single_vector_w(random_w, numpy_x, numpy_y)
    print("Error rate for random weights",error_rate_random_weights)

    # Testing train_and_evaluate
    np.random.seed(8) # to eliminate randomness
    numpy_x, numpy_y = read_csv_convert_to_numpy(fileName='carSUV_normalized.csv');
    trained_w = train_and_evaluate(numpy_x, numpy_y, n_epochs = 20, c = 0.01);
    print(trained_w)
    plot_trained_w_and_dataset(numpy_x, numpy_y, trained_w);
