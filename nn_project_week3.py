import numpy as np

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.params = {}
        self.cache = {}
        self.gradients = {}
        self.v = {}
        self.s = {}
        for i in range(1, len(layers)):
            self.params[f'W{i}'] = np.random.randn(layers[i], layers[i-1]) * np.sqrt(2/layers[i-1])
            self.params[f'b{i}'] = np.zeros((layers[i], 1))
            self.v[f'dW{i}'] = np.zeros((layers[i], layers[i-1]))
            self.v[f'db{i}'] = np.zeros((layers[i], 1))
            self.s[f'dW{i}'] = np.zeros((layers[i], layers[i-1]))
            self.s[f'db{i}'] = np.zeros((layers[i], 1))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return Z > 0

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, Z):
        s = self.sigmoid(Z)
        return s * (1 - s)

    def forward(self, X):
        A = X
        self.cache['A0'] = X
        L = len(self.layers) - 1
        for i in range(1, L):
            Z = np.dot(self.params[f'W{i}'], A) + self.params[f'b{i}']
            A = self.relu(Z)
            self.cache[f'Z{i}'] = Z
            self.cache[f'A{i}'] = A
        Z = np.dot(self.params[f'W{L}'], A) + self.params[f'b{L}']
        A = self.sigmoid(Z)
        self.cache[f'Z{L}'] = Z
        self.cache[f'A{L}'] = A
        return A

    def compute_loss(self, A, Y):
        m = Y.shape[1]
        loss = - (1/m) * np.sum(Y * np.log(A + 1e-8) + (1-Y) * np.log(1-A + 1e-8))
        return loss

    def backward(self, Y):
        L = len(self.layers) - 1
        m = Y.shape[1]
        A = self.cache[f'A{L}']
        dZ = A - Y
        for i in reversed(range(1, L + 1)):
            A_prev = self.cache[f'A{i-1}']
            dW = (1/m) * np.dot(dZ, A_prev.T)
            db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
            self.gradients[f'dW{i}'] = dW
            self.gradients[f'db{i}'] = db
            if i > 1:
                dA_prev = np.dot(self.params[f'W{i}'].T, dZ)
                dZ = dA_prev * self.relu_derivative(self.cache[f'Z{i-1}'])

    def update_sgd(self, learning_rate):
        L = len(self.layers) - 1
        for i in range(1, L + 1):
            self.params[f'W{i}'] -= learning_rate * self.gradients[f'dW{i}']
            self.params[f'b{i}'] -= learning_rate * self.gradients[f'db{i}']

    def update_adam(self, learning_rate, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
        L = len(self.layers) - 1
        for i in range(1, L + 1):
            self.v[f'dW{i}'] = beta1 * self.v[f'dW{i}'] + (1 - beta1) * self.gradients[f'dW{i}']
            self.v[f'db{i}'] = beta1 * self.v[f'db{i}'] + (1 - beta1) * self.gradients[f'db{i}']
            self.s[f'dW{i}'] = beta2 * self.s[f'dW{i}'] + (1 - beta2) * (self.gradients[f'dW{i}'] ** 2)
            self.s[f'db{i}'] = beta2 * self.s[f'db{i}'] + (1 - beta2) * (self.gradients[f'db{i}'] ** 2)
            v_dW_corrected = self.v[f'dW{i}'] / (1 - beta1 ** t)
            v_db_corrected = self.v[f'db{i}'] / (1 - beta1 ** t)
            s_dW_corrected = self.s[f'dW{i}'] / (1 - beta2 ** t)
            s_db_corrected = self.s[f'db{i}'] / (1 - beta2 ** t)
            self.params[f'W{i}'] -= learning_rate * v_dW_corrected / (np.sqrt(s_dW_corrected) + epsilon)
            self.params[f'b{i}'] -= learning_rate * v_db_corrected / (np.sqrt(s_db_corrected) + epsilon)

    def train(self, X, Y, epochs, learning_rate, optimizer='sgd', batch_size=None):
        m = X.shape[1]
        t = 0
        for epoch in range(epochs):
            if batch_size:
                permutation = np.random.permutation(m)
                X_shuffled = X[:, permutation]
                Y_shuffled = Y[:, permutation]
                for i in range(0, m, batch_size):
                    X_batch = X_shuffled[:, i:i+batch_size]
                    Y_batch = Y_shuffled[:, i:i+batch_size]
                    A = self.forward(X_batch)
                    self.backward(Y_batch)
                    t += 1
                    if optimizer == 'adam':
                        self.update_adam(learning_rate, t)
                    else:
                        self.update_sgd(learning_rate)
            else:
                A = self.forward(X)
                self.backward(Y)
                t += 1
                if optimizer == 'adam':
                    self.update_adam(learning_rate, t)
                else:
                    self.update_sgd(learning_rate)
