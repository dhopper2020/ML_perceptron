import sys
import re
from math import log
import math

MAX_ITERS = 100


# Load data from a file
def read_data(filename):
    f = open(filename, 'r')
    p = re.compile(',')
    data = []
    header = f.readline().strip()
    varnames = p.split(header)
    namehash = {}
    for l in f:
        example = [int(x) for x in p.split(l.strip())]
        x = example[0:-1]
        y = example[-1]
        # Each example is a tuple containing both x (vector) and y (int)
        data.append((x, y))
    return data, varnames


# Learn weights using the perceptron algorithm
def train_perceptron(data):
    # Initialize weight vector and bias
    numvars = len(data[0][0])
    w = [0.0] * numvars
    b = 0.0
    # initialize a
    a = 0.0

    # loop through max iters which is 100
    for i in range(MAX_ITERS):
        # go through all the examples in data
        for example in data:
            # go through all the attributes and get the sum needed, and put it into a
            for d in range(numvars):
                a += w[d]*example[0][d]
            a += b
            # if ya <= 0, update the weights and the bias values
            if example[1]*a <= 0.0:
                for d in range(numvars):
                    w[d] = w[d] + example[1]*example[0][d]
                b = b + example[1]
            # reset a so that it can be reinitialized with the sum above
            a = 0.0

    return w, b


# Compute the activation for input x.
def predict_perceptron(model, x):
    (w, b) = model
    numweights = len(w)
    a = 0
    # go through all the weights and compute the sum and put it into a
    for d in range(numweights):
        a += w[d]*x[d]
    a += b

    # if a is greater than 0, return 1 otherwise return false
    if a > 0:
        return 1.0
    else:
        return -1.0


# Load train and test data.  Learn model.  Report accuracy.
def main(argv):
    # Process command line arguments.
    # (You shouldn't need to change this.)
    if (len(argv) != 3):
        print('Usage: perceptron.py <train> <test> <model>')
        sys.exit(2)
    (train, varnames) = read_data(argv[0])
    (test, testvarnames) = read_data(argv[1])
    modelfile = argv[2]

    # Train model
    (w, b) = train_perceptron(train)

    # Write model file
    f = open(modelfile, "w+")
    f.write('%f\n' % b)
    for i in range(len(w)):
        f.write('%s %f\n' % (varnames[i], w[i]))

    # Make predictions, compute accuracy
    correct = 0
    for (x, y) in test:
        activation = predict_perceptron((w, b), x)
        print(activation)
        if activation * y > 0:
            correct += 1
    acc = float(correct) / len(test)
    print("Accuracy: ", acc)


if __name__ == "__main__":
    main(sys.argv[1:])
