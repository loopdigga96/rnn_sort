import numpy as np
import itertools


# Define the linear tensor transformation layer
class Linear(object):
    """The linear tensor layer applies a linear tensor dot product
    and a bias to its input."""

    def __init__(self, n_in, n_out, tensor_order, W=None, b=None):
        """Initialse the weight W and bias b parameters."""
        a = np.sqrt(6.0 / (n_in + n_out))
        self.W = (np.random.uniform(-a, a, (n_in, n_out))
                  if W is None else W)
        self.b = (np.zeros((n_out)) if b is None else b)
        # Axes summed over in backprop
        self.bpAxes = tuple(range(tensor_order - 1))

    def forward(self, X):
        """Perform forward step transformation with the help
        of a tensor product."""
        # Same as: Y[i,j,:] = np.dot(X[i,j,:], self.W) + self.b
        #          (for i,j in X.shape[0:1])
        # Same as: Y = np.einsum('ijk,kl->ijl', X, self.W) + self.b
        return np.tensordot(X, self.W, axes=((-1), (0))) + self.b

    def backward(self, X, gY):
        """Return the gradient of the parmeters and the inputs of
        this layer."""
        # Same as: gW = np.einsum('ijk,ijl->kl', X, gY)
        # Same as: gW += np.dot(X[:,j,:].T, gY[:,j,:])
        #          (for i,j in X.shape[0:1])
        gW = np.tensordot(X, gY, axes=(self.bpAxes, self.bpAxes))
        gB = np.sum(gY, axis=self.bpAxes)
        # Same as: gX = np.einsum('ijk,kl->ijl', gY, self.W.T)
        # Same as: gX[i,j,:] = np.dot(gY[i,j,:], self.W.T)
        #          (for i,j in gY.shape[0:1])
        gX = np.tensordot(gY, self.W.T, axes=((-1), (0)))
        return gX, gW, gB


# Define the logistic classifier layer
class LogisticClassifier(object):
    """The logistic layer applies the logistic function to its
    inputs."""

    def forward(self, X):
        """Perform the forward step transformation."""
        return 1. / (1. + np.exp(-X))

    def backward(self, Y, T):
        """Return the gradient with respect to the loss function
        at the inputs of this layer."""
        # Average by the number of samples and sequence length.
        return (Y - T) / (Y.shape[0] * Y.shape[1])

    def loss(self, Y, T):
        """Compute the loss at the output."""
        return -np.mean((T * np.log(Y)) + ((1 - T) * np.log(1 - Y)))


class SoftmaxClassifier:
    def forward(self, X, theta=1.0, axis=2):
        "Takes X as 3d tensor"

        # multiply y against the theta parameter,
        y = X * float(theta)
        # subtract the max for numerical stability
        y = y - np.expand_dims(np.max(y, axis=axis), axis)
        # exponentiate y
        y = np.exp(y)
        # take the sum along the specified axis
        ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
        # finally: divide elementwise
        return y / ax_sum

    def loss(self, Y, T):
        """
        Y is the output from fully connected layer passed through softmax (batch_size x num_examples x num_classes)
        T is labels (batch_size x num_examples x 1)
            Note that y is not one-hot encoded vector.
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        #         print(f"Y.shape {Y.shape}")
        #         print(f"T.shape {T.shape}")
        m = T.shape[1]
        # ps = self.forward(Y, axis=2)

        losses = []
        for idx, p in enumerate(Y):
            log_likelihood = -np.log(p[range(m), T[idx].flatten()])
            loss = np.sum(log_likelihood) / m
            losses.append(loss)
        return np.mean(losses)

    def backward(self, X, T):
        """
        X is the output from fully connected layer passed through softmax (batch_size x num_examples x num_classes)
        T is labels (batch_size x num_examples x 1)
            Note that y is not one-hot encoded vector.
            It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
        """
        delta = np.zeros(X.shape)
        m = T.shape[1]

        for idx in range(len(delta)):
            # x = Y[idx]
            # grad = self.forward(x, axis=1)
            # grad = x
            grad = X[idx]
            grad[range(m), T[idx].flatten()] -= 1
            grad = grad / m
            delta[idx] = grad
        return delta


# Define tanh layer
class TanH(object):
    """TanH applies the tanh function to its inputs."""

    def forward(self, X):
        """Perform the forward step transformation."""
        return np.tanh(X)

    def backward(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        gTanh = 1.0 - (Y ** 2)
        return (gTanh * output_grad)


# TODO: merge it with RNN
# Define internal state update layer
class RecurrentStateUpdate(object):
    """Update a given state."""

    def __init__(self, nbStates, W, b):
        """Initialse the linear transformation and tanh transfer
        function."""
        self.linear = Linear(nbStates, nbStates, 2, W, b)
        self.tanh = TanH()

    def forward(self, Xk, Sk):
        """Return state k+1 from input and state k."""
        return self.tanh.forward(Xk + self.linear.forward(Sk))

    def backward(self, Sk0, Sk1, output_grad):
        """Return the gradient of the parmeters and the inputs of
        this layer."""
        gZ = self.tanh.backward(Sk1, output_grad)
        gSk0, gW, gB = self.linear.backward(Sk0, gZ)
        return gZ, gSk0, gW, gB


# Define layer that unfolds the states over time
class RNN(object):
    """Unfold the recurrent states."""

    def __init__(self, nbStates, nbTimesteps):
        """Initialse the shared parameters, the inital state and
        state update function."""
        a = np.sqrt(6. / (nbStates * 2))
        self.W = np.random.uniform(-a, a, (nbStates, nbStates))
        self.b = np.zeros((self.W.shape[0]))  # Shared bias
        self.S0 = np.zeros(nbStates)  # Initial state
        self.nbTimesteps = nbTimesteps  # Timesteps to unfold
        self.stateUpdate = RecurrentStateUpdate(
            nbStates, self.W, self.b)  # State update function

    def forward(self, X):
        """Iteratively apply forward step to all states."""
        # State tensor
        S = np.zeros((X.shape[0], X.shape[1] + 1, self.W.shape[0]))
        S[:, 0, :] = self.S0  # Set initial state
        for k in range(self.nbTimesteps):
            # Update the states iteratively
            S[:, k + 1, :] = self.stateUpdate.forward(X[:, k, :], S[:, k, :])
        return S

    def backward(self, X, S, gY):
        """Return the gradient of the parmeters and the inputs of
        this layer."""
        # Initialise gradient of state outputs
        gSk = np.zeros_like(gY[:, self.nbTimesteps - 1, :])
        # Initialse gradient tensor for state inputs
        gZ = np.zeros_like(X)
        gWSum = np.zeros_like(self.W)  # Initialise weight gradients
        gBSum = np.zeros_like(self.b)  # Initialse bias gradients
        # Propagate the gradients iteratively
        for k in range(self.nbTimesteps - 1, -1, -1):
            # Gradient at state output is gradient from previous state
            #  plus gradient from output
            gSk += gY[:, k, :]
            # Propgate the gradient back through one state
            gZ[:, k, :], gSk, gW, gB = self.stateUpdate.backward(
                S[:, k, :], S[:, k + 1, :], gSk)
            gWSum += gW  # Update total weight gradient
            gBSum += gB  # Update total bias gradient
        # Get gradient of initial state over all samples
        gS0 = np.sum(gSk, axis=0)
        return gZ, gWSum, gBSum, gS0


# Define the full network
class ModelSort(object):

    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 sequence_len: int, tensor_dim: int = 3):
        tensor_dim = 3
        self.lr = 1e-3

        # Input layer
        self.input_linear = Linear(input_size, hidden_size, tensor_dim)
        # Recurrent layer
        self.rnn = RNN(hidden_size, sequence_len)
        # Linear output transform
        self.output_linear = Linear(hidden_size, output_size, tensor_dim)
        self.classifier = SoftmaxClassifier()  # Classification output
        self.sequence_len = sequence_len

    def train_on_batch(self, x_batch, y_batch):
        """
        This method makes forward run and update its parameters
        :param x_batch: 3d tensor (batch_size, seq_length, input_length)
        :param y_batch: 3d tensor (batch_size, seq_length, 1)
        :return:
        """
        linear_out, rnn_states, out, probabilities = self.forward(x_batch)
        self.backward(x_batch, probabilities, linear_out, rnn_states, y_batch)

    def forward(self, x_batch):
        """
        :param x_batch: 3d tensor (batch_size, seq_length, input_length)
        :return: linear_out, rnn_states, linear_out, probabilites
        """
        linear_out = self.input_linear.forward(x_batch)
        rnn_states = self.rnn.forward(linear_out)
        out = self.output_linear.forward(rnn_states[:, 1:self.sequence_len + 1, :])
        probabilities = self.classifier.forward(out)

        return linear_out, rnn_states, out, probabilities

    def backward(self, x_batch, y_pred, linear_out, rnn_states, y_batch):
        """
        Make computing of gradients and update parameters
        :param x_batch: 3d tensor (batch_size, seq_length, input_length)
        :param y_pred (batch_size, seq_length, number_of_classes)
        :param linear_out:
        :param rnn_states:
        :param y_batch: 3d tensor (batch_size, seq_length, 1)
        :return:
        """

        # get gradients for all layers
        gradient = self.classifier.backward(y_pred, y_batch)
        gradient, g_lout_w, g_lout_b = self.output_linear.backward(
            rnn_states[:, 1:self.sequence_len + 1, :], gradient)

        # Propagate gradient backwards through time
        gradient, g_rnn_w, g_rnn_b, g_init_state = self.rnn.backward(
            linear_out, rnn_states, gradient)

        g_x, g_lin_w, g_lin_b = self.input_linear.backward(x_batch, gradient)
        # Return the parameter gradients of: linear output weights,
        #  linear output bias, recursive weights, recursive bias, #
        #  linear input weights, linear input bias, initial state.

        gradients = [g for g in itertools.chain(
                     np.nditer(g_init_state),
                     np.nditer(g_lin_w),
                     np.nditer(g_lin_b),
                     np.nditer(g_rnn_w),
                     np.nditer(g_rnn_b),
                     np.nditer(g_lout_w),
                     np.nditer(g_lout_b))]

        # update weights
        for idx, parameter in enumerate(self.get_params_iter()):
            parameter -= self.lr * gradients[idx]

        # return gradients

    def predict_proba(self, x_batch):
        """
        :param x_batch: 3d tensor of (batch_size, seq_length, input_length)
        :return: y_proba (batch_size, seq_length, number_of_classes)
        """
        #TODO: add predict method wit argmaax
        _, _, _, y_proba = self.forward(x_batch)
        return y_proba

    # def getParamGrads(self, X, T):
    #     """Return the gradients with respect to input X and
    #     target T as a list. The list has the same order as the
    #     get_params_iter iterator."""
    #     recIn, S, Z, Y = self.forward(X)
    #     gWout, gBout, gWrec, gBrec, gWin, gBin, gS0 = self.backward(X, Y, recIn, S, T)
    #     return [g for g in itertools.chain(
    #         np.nditer(gS0),
    #         np.nditer(gWin),
    #         np.nditer(gBin),
    #         np.nditer(gWrec),
    #         np.nditer(gBrec),
    #         np.nditer(gWout),
    #         np.nditer(gBout))]

    def loss(self, y_pred, y_true):
        """
        :param y_pred: predicted probabilities (batch_size, seq_length, number_of_classes)
        :param y_true: true labels (batch_size, seq_length, 1)
        :return: Cross entropy loss over batch
        """
        return self.classifier.loss(y_pred, y_true)

    def get_params_iter(self):
        """Returns iterator over all parameters in model;
        np.nditer is efficient iterator from numpy; parameters are idetable inplace"""
        return itertools.chain(
            np.nditer(self.rnn.S0, op_flags=['readwrite']),
            np.nditer(self.input_linear.W, op_flags=['readwrite']),
            np.nditer(self.input_linear.b, op_flags=['readwrite']),
            np.nditer(self.rnn.W, op_flags=['readwrite']),
            np.nditer(self.rnn.b, op_flags=['readwrite']),
            np.nditer(self.output_linear.W, op_flags=['readwrite']),
            np.nditer(self.output_linear.b, op_flags=['readwrite']))
