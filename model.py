import numpy as np
import itertools


class TanH:
    """TanH applies the tanh function to its inputs."""

    def forward(self, x):
        return np.tanh(x)

    def backward(self, x, gradient):
        g_tanh = 1.0 - (x ** 2)
        return g_tanh * gradient


class SoftmaxClassifier:
    def forward(self, x):
        """
        :param x: 3d tensor (batch_size, seq_length, input_size)
        :return: softmax probabilities
        """
        
        x = x - np.expand_dims(np.max(x, axis=2), 2)
        exp = np.exp(x)
        exp_sum = exp.sum(-1)
        return exp / exp_sum[:,:,np.newaxis]
#         axis = 2
#         # subtract the max for numerical stability
#         y = x - np.expand_dims(np.max(x, axis=axis), axis)
#         y = np.exp(y)
#         # take the sum along the specified axis
#         ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)
#         return y / ax_sum

    def loss(self, y_pred, y_true):
        """
        Computes Cross entropy loss
        :param y_pred: softmax activations (batch_size, seq_length, number_of_classes)
        :param y_true: ground truth labels (batch_size, seq_length, 1)
        :return: mean cross entropy loss over batch
        """

        seq_length = y_true.shape[1]

        losses = []
        for idx, p in enumerate(y_pred):
            # compute log likelihood
            log_likelihood = -np.log(p[range(seq_length), y_true[idx].flatten()])
            loss = np.sum(log_likelihood) / seq_length
            losses.append(loss)

        return np.mean(losses)

    def backward(self, y_pred, y_true):
        """
        Computes gradients of loss function
        :param y_pred: softmax activations (batch_size, seq_length, number_of_classes)
        :param y_true: ground truth labels (batch_size, seq_length, 1)
        :return: gradient
        """

        delta = np.zeros(y_pred.shape)
        m = y_true.shape[1]

        for idx in range(len(delta)):
            grad = y_pred[idx]
            grad[range(m), y_true[idx].flatten()] -= 1
            # grad = grad / m
            delta[idx] = grad

        # return delta
        return delta / (y_pred.shape[0] * y_pred.shape[1])


class Linear:
    def __init__(self, input_size: int, output_size: int, tensor_dim: int,
                 weights=None, bias=None):
        a = np.sqrt(6.0 / (input_size + output_size))
        self.W = (np.random.uniform(-a, a, (input_size, output_size))
                  if weights is None else weights)
        self.b = (np.zeros(output_size) if bias is None else bias)
        # Axes summed over in backprop
        self.axes = tuple(range(tensor_dim - 1))

    def forward(self, x):
        """
        makes forward pass
        :param x: n-d tensor
        :return: linear transformation
        """
        # Same as: Y[i,j,:] = np.dot(X[i,j,:], self.W) + self.b
        #          (for i,j in X.shape[0:1])
        # Same as: Y = np.einsum('ijk,kl->ijl', X, self.W) + self.b
        return np.tensordot(x, self.W, axes=((-1), (0))) + self.b

    def backward(self, x, gradient):
        """
        :param x: n-d tensor
        :param gradient: gradient on previous step of backpropagation
        :return: gradient, g_w, g_b
        """
        # Same as: gW = np.einsum('ijk,ijl->kl', X, gY)
        # Same as: gW += np.dot(X[:,j,:].T, gY[:,j,:])
        #          (for i,j in X.shape[0:1])
        g_w = np.tensordot(x, gradient, axes=(self.axes, self.axes))
        g_b = np.sum(gradient, axis=self.axes)
        # Same as: gX = np.einsum('ijk,kl->ijl', gY, self.W.T)
        # Same as: gX[i,j,:] = np.dot(gY[i,j,:], self.W.T)
        #          (for i,j in gY.shape[0:1])
        gradient = np.tensordot(gradient, self.W.T, axes=((-1), (0)))
        return gradient, g_w, g_b


class RNNCell:

    def __init__(self, hidden_size, W, b):
        tensor_dim = 2
        self.linear = Linear(hidden_size, hidden_size, tensor_dim,
                             W, b)
        self.tanh = TanH()

    def forward(self, x, previous_state):
        """
        This function makes one forward pass
        :param x: 2d tensor (batch_size, hidden_size)
        :param previous_state: 2d tensor (batch_size x hidden_size) of rnn state on previous forward pass
        :return: rnn cell output
        """
        return self.tanh.forward(x + self.linear.forward(previous_state))

    def backward(self, previous_state, current_state, gradient_state):
        """
        :param previous_state: 2d tensor (batch_size x hidden_size) of rnn state
        :param current_state: 2d tensor (batch_size x hidden_size) of rnn state
        :param gradient_state: accumulated gradient during BPTT
        :return:gradient, gradient_state, g_w, g_b
        """

        gradient = self.tanh.backward(current_state, gradient_state)
        gradient_state, g_w, g_b = self.linear.backward(previous_state, gradient)
        return gradient, gradient_state, g_w, g_b


# Define layer that unfolds the states over time
class RNNLayer:
    """Unfold the recurrent states."""

    def __init__(self, hidden_size, sequence_length):

        a = np.sqrt(6. / (hidden_size * 2))
        self.W = np.random.uniform(-a, a, (hidden_size, hidden_size))
        self.b = np.zeros((self.W.shape[0]))
        self.rnn_cell = RNNCell(
            hidden_size, self.W, self.b)
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length

        self.initial_state = np.zeros(hidden_size)

    def forward(self, x):
        """
        :param x: 3d tensor (batch_size, seq_length, input_length)
        :return: all hidden states
        """
        states = np.zeros((x.shape[0], x.shape[1] + 1, self.hidden_size))
        states[:, 0] = self.initial_state  # Set initial state

        for idx in range(self.sequence_length):
            # Update the states iteratively
            states[:, idx + 1] = self.rnn_cell.forward(x[:, idx], states[:, idx])
        return states

    def backward(self, x, states, input_gradient):
        """
        This method computes BPTT and returns all necessary gradients
        :param x: 3d tensor (batch_size, seq_length, input_length)
        :param states: 3d tensor of all rnn states (batch_size, seq_length, hidden_size)
        :param input_gradient: gradient value on previous layers
        :return: gradient, g_w_sum, g_b_sum, g_initial_state
        """

        # Initialise gradient of state outputs
        gradient_state = np.zeros_like(input_gradient[:, self.sequence_length - 1])

        # Initialse gradient tensor for state inputs
        gradient = np.zeros_like(x)
        g_w_sum = np.zeros_like(self.W)  # Initialise weight gradients
        g_b_sum = np.zeros_like(self.b)  # Initialise bias gradients

        # Propagate the gradients iteratively
        for k in range(self.sequence_length - 1, -1, -1):
            # Gradient at state output is gradient from previous state plus gradient from output
            gradient_state += input_gradient[:, k]

            # Propagate the gradient back through one state
            gradient[:, k], gradient_state, g_w, g_b = self.rnn_cell.backward(
                states[:, k], states[:, k + 1], gradient_state)

            g_w_sum += g_w  # Update total weight gradient
            g_b_sum += g_b  # Update total bias gradient

        # Get gradient of initial state over all samples
        g_initial_state = np.sum(gradient_state, axis=0)
        return gradient, g_w_sum, g_b_sum, g_initial_state


class ModelSort:

    def __init__(self, input_size: int, output_size: int, hidden_size: int,
                 sequence_len: int, tensor_dim: int = 3):
        tensor_dim = 3
        self.lr = 1e-3

        self.input_linear = Linear(input_size, hidden_size, tensor_dim)
        self.rnn = RNNLayer(hidden_size, sequence_len)
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
        gradients = self.backward(x_batch, probabilities, linear_out, rnn_states, y_batch)

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
        :param rnn_states: 3d tensor of all rnn states (batch_size, seq_length, hidden_size)
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

        return gradients

    def predict_proba(self, x_batch):
        """
        :param x_batch: 3d tensor of (batch_size, seq_length, input_length)
        :return: y_proba (batch_size, seq_length, number_of_classes)
        """
        # TODO: add predict method wit argmaax
        _, _, _, y_proba = self.forward(x_batch)
        return y_proba

    def predict(self, x_batch):
        """
        This method predicts class
        :param x_batch: 3d tensor of (batch_size, seq_length, input_length)
        :return: 2d tensor of predictions (batch_size, seq_length)
        """
        y_proba = self.predict_proba(x_batch)
        return np.argmax(y_proba, axis=2)

    def get_gradients(self, x_batch, y_batch):
        """Return the gradients with respect to input X and
        target T as a list. The list has the same order as the
        get_params_iter iterator."""
        linear_out, rnn_states, out, probabilities = self.forward(x_batch)
        return self.backward(x_batch, probabilities, linear_out, rnn_states, y_batch)

        # recIn, S, Z, Y = self.forward(X)
        # gWout, gBout, gWrec, gBrec, gWin, gBin, gS0 = self.backward(X, Y, recIn, S, T)
        # return [g for g in itertools.chain(
        #     np.nditer(gS0),
        #     np.nditer(gWin),
        #     np.nditer(gBin),
        #     np.nditer(gWrec),
        #     np.nditer(gBrec),
        #     np.nditer(gWout),
        #     np.nditer(gBout))]

    def loss(self, y_pred, y_true):
        """
        :param y_pred: predicted probabilities (batch_size, seq_length, number_of_classes)
        :param y_true: true labels (batch_size, seq_length, 1)
        :return: Cross entropy loss over batch
        """
        return self.classifier.loss(y_pred, y_true)

    def get_params_iter(self):
        """
        Returns iterator over all parameters in model;
        np.nditer is efficient iterator from numpy; parameters are idetable inplace
        """
        return itertools.chain(
            np.nditer(self.rnn.initial_state, op_flags=['readwrite']),
            np.nditer(self.input_linear.W, op_flags=['readwrite']),
            np.nditer(self.input_linear.b, op_flags=['readwrite']),
            np.nditer(self.rnn.W, op_flags=['readwrite']),
            np.nditer(self.rnn.b, op_flags=['readwrite']),
            np.nditer(self.output_linear.W, op_flags=['readwrite']),
            np.nditer(self.output_linear.b, op_flags=['readwrite']))
