import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.n_output = n_output
        self.hidden_layer = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu_layer = ReLULayer()

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        params = self.params()	
        for param_key in params:
            params[param_key].grad = 0
		
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model

        a = self.hidden_layer.forward(X)
        output = self.relu_layer.forward(a)
        loss, dprediction = softmax_with_cross_entropy(output, y)
        d_out_hidden = self.relu_layer.backward(dprediction)
        self.hidden_layer.backward(d_out_hidden)

        # After that, implement l2 regularization on all params
        # Hint: use self.params()
        for param_key in params:
            loss += self.reg*np.sum(np.square(params[param_key].value))
            params[param_key].grad += 2*np.array(params[param_key].value)*reg_strength

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """

		
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        raise Exception("Not implemented!")
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params

        hidden_layer_params = self.hidden_layer.params()
        for param_key in hidden_layer_params:
            result[param_key] = hidden_layer_params[param_key]

        return result
