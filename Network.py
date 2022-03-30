import numpy as np


class Model:
    def __init__(self, dim1, dim2, dim3, Lambda=0.001):
        # First FC-layer
        self.fc1 = FullyConnectedLayer(dim2, dim1)
        # Second FC-layer
        self.fc2 = FullyConnectedLayer(dim3, dim2)
        # Activation layer
        # self.act = Sigmoid(dim3)
        self.act = Softmax(dim3)
        # Loss
        self.loss = CrossEntropy(dim3)
        # Regularize
        self.regularize = Regularize([self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias], Lambda=Lambda)

    def forward(self, x):
        # First FC-layer
        self.fc1.x = x.reshape(-1, 1)
        y = self.fc1.forward()
        # Second FC-layer
        self.fc2.x = y
        z = self.fc2.forward()
        # Activation layer
        self.act.x = z
        u = self.act.forward()
        return u

    def compute_loss(self, y, y_hat):
        """
        Computation of loss.
        :param y: true label
        :param y_hat: derived label
        :return: model loss + regularization loss
        """
        # Loss
        y = y.reshape((-1, 1))
        l = self.loss.compute_loss(y, y_hat) + self.regularize.compute_loss()
        return l

    def BP(self, y, y_hat, alpha):
        """
        Back propagation
        :param y: true label
        :param y_hat: derived label
        :param alpha: learning rate
        """
        y = y.reshape((-1, 1))
        grad = self.loss.gradient(y, y_hat)
        grad = self.act.BP_gradient(grad)
        grad = self.fc2.BP(grad, alpha)
        grad = self.fc1.BP(grad, alpha)
        self.regularize.BP(alpha)


class FullyConnectedLayer:
    """
    This is used for fully connected layer.
    """
    def __init__(self, dim1, dim2, coef=0.001):
        self.dim1 = dim1
        self.dim2 = dim2
        self.x = None
        self.weight = np.random.randn(dim1, dim2) * coef
        self.bias = np.random.randn(dim1, 1) * coef

    def forward(self):
        """
        :return: Wx+b
        """
        # x: vector of size = dim2 * 1
        return self.weight.dot(self.x) + self.bias

    def weight_gradient(self, x):
        """
        :param x: vector of size = dim2 * 1
        :return: gradient of weight
        """
        g = np.zeros((self.dim1, self.dim2))
        for i in range(self.dim1):
            g[i,] = x.transpose()
        return g

    def bias_gradient(self):
        """
        :return: gradient of bias
        """
        g = np.zeros((self.dim1, 1))
        return g

    def x_gradient(self):
        """
        :return: gradient of x
        """
        return self.weight

    def BP_weight_gradient(self, grad):
        """
        :param grad: gradient of last layer, vector of size = dim2 * 1
        :return: true gradient of weight
        """
        # x: vector of size = dim2 * 1
        g_last = np.zeros((self.dim1, self.dim2))
        grad = grad.reshape((self.dim1,))
        for i in range(self.dim2):
            g_last[:, i] = grad
        g_weight = self.weight_gradient(self.x)
        return np.multiply(g_last, g_weight)

    def BP_bias_gradient(self, grad):
        """
        :param grad: gradient of last layer, vector of size = dim2 * 1
        :return: true gradient of bias
        """
        g_bias = self.bias_gradient()
        return np.multiply(grad, g_bias)

    def BP_x_gradient(self, grad):
        """
        :param grad: gradient of last layer, vector of size = dim2 * 1
        :return: true gradient of x
        """
        g_last = np.zeros((self.dim1, self.dim2))
        grad = grad.reshape((self.dim1,))
        for i in range(self.dim2):
            g_last[:, i] = grad
        g_x = np.multiply(g_last, self.weight)
        return g_x.sum(0).reshape((-1, 1))

    def BP(self, grad, alpha):
        """
        Back propagation of parameter [weight] and [bias]
        :param grad: gradient of last layer, vector of size = dim2 * 1
        :param alpha: learning rate
        :return: gradient of next layer
        """
        # update weight
        g_weight = self.BP_weight_gradient(grad)
        self.weight -= alpha * g_weight
        # update bias
        g_bias = self.BP_bias_gradient(grad)
        self.bias -= alpha * g_bias
        # update next layer
        return self.BP_x_gradient(grad)


class Softmax():
    """
    This is used for activation layer.
    """
    def __init__(self, dim=10):
        self.dim = dim
        self.x = None

    def forward(self):
        x=self.x-np.max(self.x)
        e = np.exp(x)
        e /= e.sum()
        return e

    def gradient(self):
        x = self.x - np.max(self.x)
        e = np.exp(x)
        e/=e.sum()
        e = e.reshape((self.dim,))
        m = np.zeros((self.dim, self.dim))
        n = np.zeros((self.dim,self.dim))
        for i in range(self.dim):
            m[i, :] = -e
            n[:,i]=e
        grad = n * (m+np.eye(self.dim))
        return grad

    def BP_gradient(self, grad):
        """
        :param grad: gradient of last layer, vector of size = dim * 1
        :return: true gradient of x
        """
        g = self.gradient()
        return np.multiply(grad.transpose(), g).sum(1).reshape((-1,1))


class CrossEntropy():
    """
    This is used for computing loss.
    """
    def __init__(self, dim):
        self.dim = dim

    def compute_loss(self, y, y_hat):
        """
        :param y: true one-hot vector
        :param y_hat: derived score
        :return: Loss
        """
        return -(y * np.log(y_hat + 1e-6)).sum()

    def gradient(self, y, y_hat):
        """
        :param y: true one-hot vector
        :param y_hat: derived score
        :return: gradient of y_hat
        """
        return -y / (y_hat + 1e-6)



class Regularize():
    """
    This is use for L_2 regularization.
    """
    def __init__(self, parameters, Lambda=0.001):
        self.parameters = parameters
        self.Lambda = Lambda

    def compute_loss(self):
        l = 0
        for parameter in self.parameters:
            l += np.square(parameter).sum()
        return l * self.Lambda

    def BP(self, alpha):
        for parameter in self.parameters:
            parameter -= 2 * self.Lambda * alpha * parameter

# Unused activation layer
# class Sigmoid():
#     def __init__(self, dim=10):
#         self.dim = dim
#         self.x = None
#
#     def forward(self):
#         e = np.exp(self.x)
#         return e / (1 + e)
#
#     def gradient(self):
#         e = np.exp(self.x)
#         return e / np.square(1 + e)
#
#     def BP_gradient(self, grad):
#         """
#         :param grad: gradient of last layer, vector of size = dim * 1
#         :return: true gradient of x
#         """
#         g = self.gradient()
#         return np.multiply(grad, g)
