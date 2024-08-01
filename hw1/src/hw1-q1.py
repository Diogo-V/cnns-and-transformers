#!/usr/bin/env python

# Deep Learning Homework 1

import argparse
import random
import os

import numpy as np
import matplotlib.pyplot as plt

import utils


def configure_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W: np.ndarray = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X: np.ndarray, y: np.ndarray, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):

    def __init__(self, n_classes, n_features, **kwargs):
        super().__init__(n_classes, n_features, **kwargs)
        self.LEARNING_RATE = 1


    def update_weight(self, x_i: np.ndarray, y_i: np.number, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a

        # 1st -> Predicts value for this x instance (y_hat)
        y_hat_i = self.predict(x_i)
        
        # 2st -> If the predicted value is different from the correct one, we update the weights
        if y_hat_i != y_i:
            predicted = np.multiply(self.LEARNING_RATE, x_i.T)
            self.W[y_i,:] = self.W[y_i,:] + predicted
            self.W[y_hat_i,:] = self.W[y_hat_i,:] - predicted

class LogisticRegression(LinearModel):

    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b

        # Label scores according to the model (num_labels x 1).
        label_scores = self.W.dot(x_i)[:, None]
        # One-hot vector with the true label (num_labels x 1).
        y_one_hot = np.zeros((np.size(self.W, 0), 1))
        y_one_hot[y_i] = 1
        
        # Softmax function.
        # This gives the label probabilities according to the model (num_labels x 1).
        label_probabilities = np.exp(label_scores) / np.sum(np.exp(label_scores))
        # SGD update. W is num_labels x num_features.
        self.W += learning_rate * (y_one_hot - label_probabilities) * x_i[None, :]

class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().
    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.

        # n_classes -> 10
        # n_features -> 784

        units = [n_features, hidden_size + 1, n_classes]
        # # First is input size, last is output size.

        # Initialize all weights and biases randomly.
        self.W1 = np.random.normal(0.1, 0.1, (units[1], units[0]))
        self.b1 = np.zeros(units[1])
        
        self.W2 = np.random.normal(0.1, 0.1, (units[2], units[1]))
        self.b2 = np.zeros(units[2])

        self.weights = [self.W1, self.W2]
        self.biases = [self.b1, self.b2]
        self.activation_func = [self.relu, self.softmax]

    def relu(self, val):
        def inner(out):
            return max(0.0, out)
        return np.vectorize(inner)(val)

    def softmax(self, vals):
        f = vals - np.max(vals)
        return np.exp(f) / np.sum(np.exp(f))

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        
        results = []
        for x in X:
            
            z1 = self.W1.dot(x.T) + self.b1
            a1 = np.vectorize(self.relu)(z1)

            z2 = self.W2.dot(a1.T) + self.b2
            a2 = self.softmax(z2)
            t = np.argmax(a2)

            results.append(t)

        return results

    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        # print(f"y_hat size: {len(y_hat)}")
        # print(f"y_hat: {y_hat}")
        # print(f"y size: {len(y)}")
        # print(f"y: {y}")
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def update_parameters(self, weights, biases, grad_weights, grad_biases, eta):
        num_layers = len(weights)
        for i in range(num_layers):
            weights[i] -= eta*grad_weights[i]
            biases[i] -= eta*grad_biases[i]

    def relu_derivative(self, dx):
        def inner(out):
            return 1 if out > 0 else 0
        return np.vectorize(inner)(dx)

    def forward(self, x, weights, biases):
        num_layers = len(weights)
        result_a = []
        for i in range(num_layers):
            h = x if i == 0 else result_a[i-1]
            z = weights[i].dot(h) + biases[i]
            if i < num_layers-1:  # relu for hidden layer and CE (softmax) for output layer
                result_a.append(self.relu(z))
            else:
                result_a.append(self.softmax(z))
        # For classification this is a vector of logits (label scores).
        # For regression this is a vector of predictions.
        return result_a[-1], result_a

    def cross_entropy_derivative(self, out, target):
        # return self.softmax(out) - target
        t = np.zeros(out.shape)
        t[target] = 1
        return out - t

    def backward(self, x, y, output, hiddens, weights):
        num_layers = len(weights)
        
        # Grad of loss wrt last z cross entropy
        grad_z = self.cross_entropy_derivative(output, y)

        grad_weights = []
        grad_biases = []
        for i in range(num_layers-1, -1, -1):
            # Gradient of hidden parameters.
            h = x if i == 0 else hiddens[i-1]
            
            grad_weights.append(grad_z[:, None].dot(h[:, None].T))
            grad_biases.append(grad_z)

            # Gradient of hidden layer below.
            grad_h = weights[i].T.dot(grad_z)

            # Gradient of hidden layer below before activation.
            grad_z = grad_h * self.relu_derivative(h)   # Grad of loss wrt z3.

        grad_weights.reverse()
        grad_biases.reverse()
        return grad_weights, grad_biases


    def train_epoch(self, X: np.ndarray, y: np.ndarray, learning_rate=0.001):
        for x_i, y_i in zip(X, y):
            output_a, results_a = self.forward(x_i, self.weights, self.biases)
            grad_weights, grad_biases = self.backward(x_i, y_i, output_a, results_a, self.weights)
            self.update_parameters(self.weights, self.biases, grad_weights, grad_biases, learning_rate)


def plot(epochs, valid_accs, test_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.plot(epochs, valid_accs, label='validation')
    plt.plot(epochs, test_accs, label='test')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-layers', type=int, default=1,
                        help="""Number of hidden layers (needed only for MLP,
                        not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_classification_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]

    n_classes = np.unique(train_y).size  # 10
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    valid_accs = []
    test_accs = []
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        model.train_epoch(
            train_X,
            train_y,
            learning_rate=opt.learning_rate
        )
        valid_accs.append(model.evaluate(dev_X, dev_y))
        test_accs.append(model.evaluate(test_X, test_y))

    # plot
    plot(epochs, valid_accs, test_accs)


if __name__ == '__main__':
    main()
