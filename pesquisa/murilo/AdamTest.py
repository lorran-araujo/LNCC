
import os
import sys
import numpy as np
import utils as ut
import pandas as pd
import timeit
import matplotlib.pyplot as plt

# start_time = timeit.default_timer()

# -------------------------------------------------------------------------

# -------------------------------------------------------------------------

class NeuralNet:

    def fit(self, X_train, y_train, layers, alpha, epochs):

        """
        Main function that trans/fits the neural net by running through multiple epochs
        :param X_train: training data feautures
        :param y_train: training data labels; y_actual
        :param layers: number of hidden layers and neurons in each of them
        :param alpha: learning rate initailized value; later changed by adam optimizer
        :param epochs: number of iterations to be run
        :return: trained neural net parameters, to be used for predicting on test data
        """

        # Initialize Weights and Biases
        parameters = self.initialize(layers)

        cost_function, learning_curve = [], []

        # Binarize Labels
        classes = list(set(y_train))
        y_bin = ut.label_binarize(y_train, classes)  # Y = [1,0,0,0], [0,1,0,0]...

        for j in range(epochs):

            # Making batches of data
            # X = X_train
            # Y = y_train
            #
            # batch_slice = ut.gen_batches(X_train.shape[0], X_train.shape[0] / 1000)
            # for i in batch_slice:
            #     X_train = X[int(i.start):int(i.stop) + 1]
            #     y_train = Y[int(i.start):int(i.stop) + 1]

            # ---------------------------------------------------
            y_hat, parameters = self.forward_prop(X_train, parameters, layers)
            # y_hat is the predicted label probability by softmax

            # Utils: log_loss(y_true, y_prob)
            log_loss = ut.log_loss(y_bin, y_hat)

            # Back Propagation
            parameters = self.back_prop(X_train, y_bin, parameters, layers)

            # Prep variables for adam optimizer
            params, grads = self.prep_vars(layers, parameters)

            # Initialize constructor of adam optimizer
            learning_rate_init = alpha
            optimizer = AdamOptimizer(params, learning_rate_init)

            # updates weights with grads
            params = optimizer.update_params(grads)  # update weights

            # Unpack results from Adam Optimizer
            parameters = self.params_unpack(params, parameters, layers)

            # Append log loss, to plot curve later
            cost_function.append(log_loss)

            # ---------------------------------------------------
            # Mapping
            if j == 0:
                class_dict = dict()
                for i in range(len(y_bin)):
                    class_dict[str(y_bin[i])] = y_train[i]
            # ---------------------------------------------------

        # Making pyplots
        # print("cost_function 0, -1", cost_function[0], cost_function[-1])
        # print("learning_curve 0, -1", learning_curve[0], learning_curve[-1])
        #
        # learning_curve = pd.DataFrame(learning_curve)
        # learning_curve.to_csv("learning_curve_blackbox21.csv", header=False, index=False)
        # print("learning_curve", learning_curve)
        #
        # plt.plot(learning_curve)
        # plt.title("Learning Curve")
        # plt.xlabel("Epochs")
        # plt.ylabel("Accuracy")
        # plt.show()
        #
        # cost_function = pd.DataFrame(cost_function)
        #
        # plt.plot(cost_function)
        # plt.title("Logistic Loss")
        # plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        # plt.show()

        return parameters, class_dict

    def prep_vars(self, layers, parameters):

        params, grads = [], []

        for j in "W", "B":
            for i in range(1, len(layers)):
                params.append(parameters[j + str(i)])
                grads.append(parameters['d' + j + str(i)])

        params = np.asarray(params)
        grads = np.asarray(grads)

        return params, grads

    def params_unpack(self, params, parameters, layers):

        """
        Unpack the results from Adam Optimizer
        :param params: Results of Adam Optimizer
        :param parameters: Original Dictionary
        :param layers: Number of layers and neurons
        :return:
        """

        j = 0
        for i in range(1, len(layers)):
            parameters['W' + str(i)] = params[j]
            j += 1

        for i in range(1, len(layers)):
            parameters['B' + str(i)] = params[j]
            j += 1

        return parameters

    def initialize(self, layers):
        """
        Xavier Initialization of weights and biases
        :param layers:
        :return: Initialized weights and biases
        """
        # We will use a dictionary throughout the code to refer to values
        parameters = {}

        # Random seed
        rand_state = np.random.RandomState(42)

        for i in range(1, len(layers)):
            bound = np.sqrt(6. / (layers[i - 1] + layers[i]))
            parameters['W' + str(i)] = rand_state.uniform(-bound, bound, (layers[i - 1], layers[i]))
            parameters['B' + str(i)] = rand_state.uniform(-bound, bound, layers[i])

        return parameters

    def forward_prop(self, data, parameters, layers):

        """
        Forward propogate values
        :param data: Input data (train and then test)
        :param parameters: Dictionary with initialized weeights and biases
        :param layers: Number of layers and neurons
        :return:
        """
        """
        Z[l] = A[l-1] * W[l] + B[l]
        A[l] = G[l](Z[l])
        """
        # A0 initialized here and not in init function, so we can run the training data through forward prop
        parameters['A' + str(0)] = data

        for i in range(1, len(layers)):
            parameters['Z' + str(i)] = np.add(np.dot(parameters['A' + str(i - 1)], parameters['W' + str(i)]),
                                              parameters['B' + str(i)])

            if i != len(layers) - 1:
                parameters['A' + str(i)] = ut.relu(parameters['Z' + str(i)])
            else:
                # Final Activation is Softmax
                parameters['A' + str(i)] = ut.softmax(parameters['Z' + str(i)])

        return parameters['A' + str(len(layers) - 1)], parameters

    def back_prop(self, X_train, Y, parameters, layers):

        """
        Back popogate error/loss to each layer (proof/logic below)
        :param X_train: Training Data
        :param Y: True lable values of trainig data
        :param parameters: Dictionary with initialized weeights and biases
        :param layers: Number of layers and neurons
        :return:
        """

        """
        Compute Derivatives: dCost/dw[l], dCost/db[l]
        Weights:
        dCost/dw[l] = dz[l]/dw[l] * da[l]/dz[l] * dCost/da[l]
            as, Z[l] = A[l-1] * W[l] + B[l]
        so, dz[l]/dw[l] = A[l-1]
            as, A[l-1] = g(Z[l])
        so, da[l]/dz[l] = g'(Z[l])
            as Cost = (A[l] - y_true)^2
        so, dCost/da[l] = 2*(A[l] - y_true)
        => dCost/dw[l] = A[l-1] * g'(Z[l]) * 2*(A[l] - y_true)
        Bias:
        dCost/db[l] = dz[l]/db[l] * da[l]/dz[l] * dCost/da[l]
            as, Z[l] = A[l-1] * W[l] + B[l]
        so, dz[l]/db[l] = 1
        => dCost/db[l] = 1 * g'(Z[l]) * 2*(A[l] - y_true)
        Since, this is for 1 example =>
        for n examples = 1/n * derivative
        """
        m = X_train.shape[0]  # Number of values; used for averaging

        parameters['dZ' + str(len(layers) - 1)] = (1 / m) * (parameters['A' + str(len(layers) - 1)] - Y)
        parameters['dW' + str(len(layers) - 1)] = np.dot(np.transpose(parameters['A' + str(len(layers) - 2)]),
                                                         parameters['dZ' + str(len(layers) - 1)])
        parameters['dB' + str(len(layers) - 1)] = parameters['dZ' + str(len(layers) - 1)].sum()

        for i in range(len(layers) - 2, 0, -1):
            parameters['dZ' + str(i)] = (1 / m) * (np.dot(parameters['dZ' + str(i + 1)],
                                                          np.transpose(parameters['W' + str(i + 1)])) *
                                                   (self.relu_derivative(parameters['Z' + str(i)])))
            parameters['dW' + str(i)] = np.dot(np.transpose(parameters['A' + str(i - 1)]), parameters['dZ' + str(i)])
            parameters['dB' + str(i)] = parameters['dZ' + str(i)].sum()

        return parameters

    def relu_derivative(self, x):
        """
        Derivative of ReLU Activation
        :param x: input data
        :return: 0 if value <= 0, else 1
        """
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def predict(self, X_test, parameters, class_dict, layers):

        """
        Call forward prop on test data and return predicted classes
        :param X_test: Test Data
        :param parameters: Neural Net Fitted Values
        :param class_dict: Mapping
        :param layers: Number of Layers and Neurons
        :return:
        """
        # Call forward Prop
        y_test, parameters = self.forward_prop(X_test, parameters, layers)

        # Binarize probabilities
        y_test = (y_test == y_test.max(axis=1)[:, None]).astype(float)

        res = []
        # Map binarized probabilities to relevant classes
        for i in range(y_test.shape[0]):
            res.append(class_dict[str(y_test[i])])

        return y_test, res


# -------------------------------------------------------------------------
class AdamOptimizer():
    """Stochastic gradient descent optimizer with Adam
    Note: All default values are from the original Adam paper
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights
    beta_1 : float, optional, default 0.9
        Exponential decay rate for estimates of first moment vector, should be
        in [0, 1)
    beta_2 : float, optional, default 0.999
        Exponential decay rate for estimates of second moment vector, should be
        in [0, 1)
    epsilon : float, optional, default 1e-8
        Value for numerical stability
    Attributes
    ----------
    learning_rate : float
        The current learning rate
    t : int
        Timestep
    ms : list, length = len(params)
        First moment vectors
    vs : list, length = len(params)
        Second moment vectors
    References
    ----------
    Kingma, Diederik, and Jimmy Ba.
    "Adam: A method for stochastic optimization."
    arXiv preprint arXiv:1412.6980 (2014).
    Example:
    ----------
    n_layers = 3
    layer_units = [10,20,5]
    # Initialize coefficient(weights) and intercept(bias) for each layer
    coefs_ = []
    intercepts_ = []
    for i in range(n_layers_ - 1):
        coef_init, intercept_init = init_coef(layer_units[i],layer_units[i + 1])  # init_coef is a function to initialize weights and bias
        coefs_.append(coef_init)
        intercepts_.append(intercept_init)
    params = coefs_ + intercepts_
    # create optimizer
    optimizer = AdamOptimizer(params, learning_rate_init = 0.001)
    # initialize grads for weights and bias
    coef_grads, intercept_grads = initialize_grads()  // function to creat numpy array to store grads
    # initialize loss for each layers
    deltas[-1] = activations[-1] - y_true////
    # backpropagation to compute grads
    coef_grads, intercept_grads = backprop(X[batch_slice], y[batch_slice], coef_grads, intercept_grads)
    grads = coef_grads + intercept_grads
    # updates weights with grads
    optimizer.update_params(grads)    //update weights
    """

    def __init__(self, params, learning_rate_init=0.001, beta_1=0.9,
                 beta_2=0.999, epsilon=1e-8):
        self.params = [param for param in params]
        self.learning_rate_init = float(learning_rate_init)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.t = 0
        self.ms = [np.zeros_like(param) for param in params]
        self.vs = [np.zeros_like(param) for param in params]

    def update_params(self, grads):
        """Update parameters with given gradients
        Parameters
        ----------
        grads : list, length = len(params)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        """
        updates = self._get_updates(grads)
        for param, update in zip(self.params, updates):
            param += update

        return self.params

    def _get_updates(self, grads):
        """Get the values used to update params with given gradients
        Parameters
        ----------
        grads : list, length = len(coefs_) + len(intercepts_)
            Containing gradients with respect to coefs_ and intercepts_ in MLP
            model. So length should be aligned with params
        Returns
        -------
        updates : list, length = len(grads)
            The values to add to params
        """
        self.t += 1
        self.ms = [self.beta_1 * m + (1 - self.beta_1) * grad
                   for m, grad in zip(self.ms, grads)]
        self.vs = [self.beta_2 * v + (1 - self.beta_2) * (grad ** 2)
                   for v, grad in zip(self.vs, grads)]
        self.learning_rate = (self.learning_rate_init *
                              np.sqrt(1 - self.beta_2 ** self.t) /
                              (1 - self.beta_1 ** self.t))
        updates = [-self.learning_rate * m / (np.sqrt(v) + self.epsilon)
                   for m, v in zip(self.ms, self.vs)]
        return updates


# -------------------------------------------------------------------------
class ReadData:

    """
    Class to Read Input Data, Test Data, and Generate Ouput Data
    """

    def inputdata(self):

        """
        Reads Input Data and Calls Neural Net to fit the data
        :return: The fitted neural net weights and biases
        """
        train_file = "datasets/cancer.csv"

        X_train, y_train = [], []

        with open(train_file, "r") as input_csv:
            data = input_csv.readlines()

            for row in range(len(data)):
                val = list(map(int, data[row].rstrip("\n").split(",")))
                X_train.append(val[:-1])
                y_train.append(val[-1])

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        # ---------------------------------------------------------------------
        obj = NeuralNet()

        input_layer = X_train[0].shape[0]  # Number of attributes
        output_layer = np.unique(y_train).size  # Number of classes

        """
        1. Specify the number of hidden layers you want
        2. Specify the number of neurons in each layer
        3. Give the initial learning rate (changed by Adam optimizer)
        4. Specify number of epochs
        5. I/P layers has neurons = Number of attributes
        5. O/P layer has neurons = Number of Classes
        """
        layers = [input_layer, 12, 12, output_layer]
        parameters, class_dict = obj.fit(X_train, y_train, layers, 0.001, 200)

        return parameters, class_dict, layers

    # -------------------------------------------------------------------------

    def testData(self, parameters, class_dict, layers):

        """
        Read Test Data and call classify function
        :param parameters: Trained Neural Net data
        :param class_dict: Mapping of binarized variables
        :param layers: Number of hidden layers, and neurons in them
        :return: N/A -> Call classify function
        """
        global test_file

        X_test = []
        with open(test_file, "r") as test_csv:
            test_data = test_csv.readlines()

            for row in range(len(test_data)):
                val = list(map(int, test_data[row].rstrip("\n").split(",")))
                X_test.append(val)

        X_test = np.asarray(X_test)

        self.classify(X_test, parameters, class_dict, layers)

    # ------------------------------------------------------

    def classify(self, X_test, parameters, class_dict, layers):

        """
        Call function to run forward prop with test data
        :param X_test: Test Data
        :param parameters: Dictionary with Trained Neural Net Weights and Biases
        :param class_dict: Mpping of Binarized Variables to Classes
        :param layers: Number of hidden layers, and neurons in them
        :return: N/A -> Print subission file
        """

        obj2 = NeuralNet()
        y_test, res = obj2.predict(X_test, parameters, class_dict, layers)
        res = np.asarray(res)

        # Create Submission
        submission = pd.DataFrame(res)

        global test_path
        global dirpath
        submission.to_csv(test_path[:10] + "_predictions.csv", header=False, index=False)

        # print(timeit.default_timer() - start_time)
    # ------------------------------------------------------

# -------------------------------------------------------------------------


obj = ReadData()
parameters, class_dict, layers = obj.inputdata()
obj.testData(parameters, class_dict, layers)