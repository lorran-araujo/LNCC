import numpy as np

'''
==================================================================================================
Helper functions for numerical functions.
We give you 4 useful functions to calculate activation and loss
1.  relu activation functions
2.  gradient of relu activation functions
3.  softmax activation functions for output layer
4.  log cross-entropy loss functions
===================================================================================================
'''
def relu(X):
    """Compute the rectified linear unit function inplace.
    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The input data.
    Returns
    -------
    X_new : {array-like, sparse matrix}, shape (n_samples, n_features)
        The transformed data.
    
    Example:
    -------
    >>> relu([-1,0,1])
    >>> [0,0,1]
    """
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


def softmax(X):
    """Compute the K-way softmax function inplace.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features) The input data.
    Returns
    -------
    X_new : array-like, shape (n_samples, n_features) The transformed data.
    
    Example:
    -------
    >>> softmax(np.array([[2.0,2.5,3.0],[1.0,1.5,4.0]]))
    >>> array([[0.18632372, 0.30719589, 0.50648039],[0.04398648, 0.07252145, 0.88349207]])
    """
    tmp = X - X.max(axis=1)[:, np.newaxis]
    np.exp(tmp, out=X)
    X /= X.sum(axis=1)[:, np.newaxis]

    return X

def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.
    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.
    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.
    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    
    Example:
    ----------
    >>> Z = np.array([[1,0,1],[0,1,2]])
    >>> delta = np.array([[0.1,0.2,0.3],[0.1,0.2,0.3]])
    >>> inplace_relu_derivative(Z, delta))
    >>> print(delta)
    >>> [[0.1 0.  0.3],[0.  0.2 0.3]]
    """
    delta[Z == 0] = 0

def log_loss(y_true, y_prob):
    """Compute Logistic loss for classification.
    Parameters
    ----------
    y_true : array-like or label indicator matrix
        Ground truth (correct) labels.
    y_prob : array-like of float, shape = (n_samples, n_classes)
        Predicted probabilities, as returned by a classifier's
        predict_proba method.
    Returns
    -------
    loss : float
        The degree to which the samples are correctly predicted.
    Example:
    -------
    >>>> log_loss(np.array([[1,1,1],[1,2,3]]), np.array([[1,1,1],[1,2,3]]))
    >>>> -2.34106561356211
    """
    if y_prob.shape[1] == 1:
        y_prob = np.append(1 - y_prob, y_prob, axis=1)

    if y_true.shape[1] == 1:
        y_true = np.append(1 - y_true, y_true, axis=1)

    return - np.multiply(y_true, np.log(y_prob)).sum() / y_prob.shape[0]

'''
===================================================================================================
Helper Function for optimizer
We give you 2 Optimizer 
1. stochastic gradient descent with momentum
2. adam
===================================================================================================
'''
class SGDOptimizer():
    """Stochastic gradient descent optimizer with momentum
    Parameters
    ----------
    params : list, length = len(coefs_) + len(intercepts_)
        The concatenated list containing coefs_ and intercepts_ in MLP model.
        Used for initializing velocities and updating params
    learning_rate_init : float, optional, default 0.1
        The initial learning rate used. It controls the step-size in updating
        the weights
    momentum : float, optional, default 0.9
        Value of momentum used, must be larger than or equal to 0
    nesterov : bool, optional, default True
        Whether to use nesterov's momentum or not. Use nesterov's if True
    Attributes
    ----------
    learning_rate : float
        the current learning rate
    velocities : list, length = len(params)
        velocities that are used to update params
    
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
    deltas[-1] = activations[-1] - y_true
    # backpropagation to compute grads
    coef_grads, intercept_grads = backprop(X[batch_slice], y[batch_slice], coef_grads, intercept_grads)
    grads = coef_grads + intercept_grads
    
    # updates weights with grads
    optimizer.update_params(grads)    //update weights
    """

    def __init__(self, params, learning_rate_init=0.1,
                 momentum=0.9, nesterov=True):
        self.params = [param for param in params]
        self.learning_rate = float(learning_rate_init)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocities = [np.zeros_like(param) for param in params]

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
        updates = [self.momentum * velocity - self.learning_rate * grad
                   for velocity, grad in zip(self.velocities, grads)]
        self.velocities = updates

        if self.nesterov:
            updates = [self.momentum * velocity - self.learning_rate * grad
                       for velocity, grad in zip(self.velocities, grads)]

        return updates


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


'''
===================================================================================================
Helper Function for generate batches
It is a Generator so you can just iterate it in for loop, see Example below:
Examples
--------
>>> from utils import gen_batches(n_samples, batch_size)
        X_batch = X[batch_slice]
        y_batch = y[batch_slice]
===================================================================================================
'''

def gen_batches(n, batch_size, min_batch_size=0):
    """Generator to create slices containing batch_size elements, from 0 to n.
    The last slice may contain less than batch_size elements, when batch_size
    does not divide n.
    Parameters
    ----------
    n : int
    batch_size : int
        Number of element in each batch
    min_batch_size : int, default=0
        Minimum batch size to produce.
    Yields
    ------
    slice of batch_size elements
    
    Examples
    --------
    >>> from utils import gen_batches(n_samples, batch_size)
        X_batch = X[batch_slice]
        y_batch = y[batch_slice]
    """
    start = 0
    for _ in range(int(n // batch_size)):
        end = start + batch_size
        if end + min_batch_size > n:
            continue
        yield slice(start, end)
        start = end
    if start < n:
        yield slice(start, n)


'''
===================================================================================================
Helper function to calculate accuracy score
Examples
--------
>>> from utils import accuracy_score
>>> accuracy_score([1,0,1,0], [1,1,1,1])
0.5
===================================================================================================
'''
def accuracy_score(y_true, y_pred):
    """Accuracy classification score.
    Parameters
    ----------
    y_true : 1d array-like
    y_pred : 1d array-like Predicted labels, as returned by a classifier.
    Returns
    -------
    score : float
    Example:
    -------
    >>> from utils import accuracy_score
    >>> accuracy_score([1,0,1,0], [1,1,1,1])
    0.5
    """

    # Compute accuracy for each possible representation
    assert y_true.shape == y_pred.shape

    score = y_true == y_pred

    return np.average(score)

'''
===================================================================================================
Helper function to transform (N_sample,) label to multi-class binary labels (N_sample, n_classes)
Examples
--------
>>> from utils import label_binarize
>>> label_binarize([1, 6], classes=[1, 2, 4, 6])
array([[1, 0, 0, 0],
       [0, 0, 0, 1]])
===================================================================================================
'''
def label_binarize(y, classes):
    """Transform multi-class labels with Dimension (N,) to
        binary labels ndarray with Dimension (N, n_classes)
       Parameters
       ----------
       y : array of shape [n_samples,]
        Target values. For example [0,1,0,2,3,4]
       classes : array-like of shape [n_classes]
        Uniquely holds the label for each class.
       neg_label : int (default: 0)
        Value with which negative labels must be encoded.
       pos_label : int (default: 1)
         Value with which positive labels must be encoded.
       Returns
       -------
       Y : numpy array of shape [n_samples, n_classes]
       Examples:
       -------
       >>> from utils import label_binarize
       >>> label_binarize([1, 6], classes=[1, 2, 4, 6])
       array([[1, 0, 0, 0],[0, 0, 0, 1]])
    """

    n_samples = y.shape[0]
    n_classes = len(classes)
    classes = np.asarray(classes)
    sorted_class = np.sort(classes)

    # binarizer label
    Y = np.zeros((n_samples, n_classes))
    indices = np.searchsorted(sorted_class, y)
    Y[np.arange(n_samples), indices] = 1

    return Y