from mlp_utils import *

class SigmoidActivation:
    """
    Sigmoid class with cost function and error
    """
    @staticmethod
    def fn(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def delta(x):
        sig = SigmoidActivation.fn(x)
        return sig * (1 - sig)

class ReLUActivation:
    """
    RelU class with cost function and error
    """
    @staticmethod
    def fn(x):
        return x * (x > 0)

    @staticmethod
    def delta(x):
        return 1. * (x > 0)

class CrossEntropyCost:
    """
    Cross Entropy class with cost function and error
    """
    @staticmethod
    def fn(a,y):
        return np.mean(np.nan_to_num( -y * np.log(a) - (1-y) * np.log(1-a)), axis=0)

    @staticmethod
    def delta(a,y):
        return (a-y)


class MeanSquareRootCost:
    """
    Mean Root Square Error class with cost function and error
    """
    @staticmethod
    def fn(a, y):
        return np.mean(np.sum((a - y) ** 2 / 2, axis = 1), axis = 0)

    @staticmethod
    def delta(a, y):
        return a - y

class SoftmaxActivation:

    @staticmethod
    def fn(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def delta(X, y):
        m = y.shape[0]
        p = softmax(X)
        dx = np.mean(p[range(m), y] - 1)
        return dx

class MultiPerceptron():

    def __init__(self, input_size, output_size, hidden_structure=[], cost_function=CrossEntropyCost, output_function=SoftmaxActivation, hidden_activation=SigmoidActivation):
        #Une liste avec la sructure de notre réseau de neurones
        self.layers = [input_size] + hidden_structure + [output_size]
        self.cost_f = cost_function()
        self.hidden_a = hidden_activation
        self.output_f = output_function
        self.initialize_weights()

    def initialize_weights(self):
        #Des paramètres intrinsèques de notre réseau de neurones
        self.weights = []
        self.biais = []
        #Initialisation des paramètres du réseau de neurones
        for i in range(len(self.layers) - 1):
            self.weights.append(2 * np.random.random((self.layers[i], self.layers[i + 1])) - 1)
            self.biais.append(np.zeros(self.layers[i + 1]))

    def prediction(self, X):
        A = X
        # Applications succéssives des multiplications matricielles
        # et des transformations non linéaire sur le vecteur d'entrée
        for i in range(len(self.layers) - 1):
            A = (self.hidden_a).fn(self.biais[i] + np.dot(A, self.weights[i]))
        return A

    def accuracy(self, X_test, y_test):
        return np.round(100*np.mean(np.equal(np.argmax(self.prediction(X_test), axis=1), np.argmax(y_test, axis=1))), 2)

    def backpropagation(self, X, Y):

        # Les gradients de chaque paramètre du réseau de neurones
        delta_W, delta_B = [], []

        # Propagation avant avec mémoire (Le réseau prédit le vecteur de sortie)
        sums, activations = [], [X]
        A = X

        for i in range(len(self.layers) - 1):
            Z = self.biais[i] + np.dot(A, self.weights[i])
            A = (self.hidden_a).fn(Z)
            activations.append(A)
            sums.append(Z)
        # Propagation arrière (Calcul et rétropropagation des erreurs commises par les paramètres du réseau)

        # Sur la dernière couche
        delta = (self.cost_f).delta(activations[-1], Y) * (self.hidden_a).delta(sums[-1])
        dW = np.mean(np.einsum('bi, bj -> bij',activations[-2], delta), axis=0)
        dB = np.mean(delta, axis=0)
        delta_W.append(dW)
        delta_B.append(dB)

        # Sur les autres couches
        for i in range(len(self.layers) - 3, -1, -1):
            delta = np.dot(delta, self.weights[i + 1].transpose()) * (self.hidden_a).delta(sums[i])
            dW = np.mean(np.einsum('bi, bj -> bij',activations[i], delta), axis=0)
            dB = np.mean(delta, axis=0)
            delta_W.append(dW)
            delta_B.append(dB)

        # Puisque les erreurs étaient ajoutées inversement, on inverse leurs vecteurs à la fin
        return delta_W[::-1], delta_B[::-1]

    def get_minibatches(self, X, y, minibatch_size, shuffleTag=True):
        m = X.shape[0]
        minibatches = []
        if shuffleTag:
            X, y = shuffle_data(X, y)
        for i in range(0, m, minibatch_size):
            X_batch = X[i:i + minibatch_size, :]
            y_batch = y[i:i + minibatch_size]
            minibatches.append((X_batch, y_batch))
        return minibatches

    def vanilla_update(self, delta_W, delta_B, alpha=0.01):
        for i in range(len(self.layers)-1):
            # Mise à jour du réseau
            self.weights[i] -= alpha * delta_W[i]
            self.biais[i] -= alpha * delta_B[i]

    def sgd(self, X_train, y_train, minibatch_size, epochs, alpha=1, verbose=True, X_test=None, y_test=None):
        minibatches = self.get_minibatches(X_train, y_train, minibatch_size)
        test_acc_log = []
        for i in range(epochs):
            loss = 0
            if verbose:
                print("")
                print("Epoch : {0}".format(i + 1))
            for X_mini_batch, y_mini_batch in minibatches:
                if (len(test_acc_log)+1)%10 == 0:
                    sys.stdout.write('\r')

                delta_W, delta_B = self.backpropagation(X_mini_batch, y_mini_batch)
                self.vanilla_update(delta_W, delta_B, alpha=alpha)
                test_acc_log += [self.accuracy(X_test, y_test)]

                if len(test_acc_log)%10 == 0:
                    sys.stdout.write("Minibatch : {0}| Test Accuracy = {1}".format(len(test_acc_log), test_acc_log[-1]))

            if verbose:
                train_acc = self.accuracy(X_train, y_train)
                test_acc = self.accuracy(X_test, y_test)
                train_loss = (self.cost_f).fn(self.prediction(X_train), y_train)
                test_loss = (self.cost_f).fn(self.prediction(X_test), y_test)
                print("")
                print("Training Loss = {0} | Training Loss = {1} | Training Accuracy = {2} | Test Accuracy = {3}".format(train_loss, test_loss, train_acc, test_acc))

        return test_acc_log

    def momentum_update(self, w_velocity, b_velocity, delta_W, delta_B, alpha=1, mu=0.9):
        for i in range(len(self.layers) - 1):
            # Mise à jour du réseau
            w_velocity[i] = mu * w_velocity[i] + alpha * np.mean(delta_W[i], axis = 0)
            b_velocity[i] = mu * b_velocity[i] + alpha * np.mean(delta_B[i], axis = 0)
            self.weights[i] -= w_velocity[i]
            self.biais[i] -= b_velocity[i]

    def sgd_momentum(self, X_train, y_train, minibatch_size, epochs, alpha=1, mu=0.9, verbose=True, X_test=None, y_test=None, nesterov=True):
        minibatches = self.get_minibatches(X_train, y_train, minibatch_size)
        for i in range(epochs):
            loss = 0
            w_velocity = [np.zeros_like(weights) for weights in self.weights]
            b_velocity = [np.zeros_like(biais) for biais in self.biais]
            if verbose:
                print("Epoch {0}".format(i + 1))
            for X_mini_batch, y_mini_batch in minibatches:
                if nesterov:
                    for i in range(len(self.weights)):
                        self.weights[i] += mu * w_velocity[i]
                        self.biais[i] += mu * b_velocity[i]
                delta_W, delta_B = self.backpropagation(X_mini_batch, y_mini_batch)
                self.momentum_update(w_velocity, b_velocity, delta_W, delta_B, alpha=alpha, mu=mu)
            if verbose:
                train_acc = self.accuracy(X_train, y_train)
                test_acc = self.accuracy(X_test, y_test)
                train_loss = (self.cost_f).fn(self.prediction(X_train), y_train)
                test_loss = (self.cost_f).fn(self.prediction(X_test), y_test)
                print("Training Loss = {0} | Training Loss = {1} | Training Accuracy = {2} | Test Accuracy = {3}".format(train_loss, test_loss, train_acc, test_acc))

    def adagrad_update(self, w_cache, b_cache, delta_W, delta_B, alpha=1):
        for i in range(len(self.layers) - 1):
            # Mise à jour du réseau
            w_cache[i] += np.mean(delta_W[i], axis=0)**2
            b_cache[i] += np.mean(delta_B[i], axis=0)**2
            self.weights[i] -= alpha * np.mean(delta_W[i], axis=0) / (np.sqrt(w_cache[i]) + 1e-8)
            self.biais[i] -= alpha * np.mean(delta_B[i], axis=0) / (np.sqrt(b_cache[i]) + 1e-8)

    def adagrad(self, X_train, y_train, minibatch_size, epochs, alpha=1, verbose=True, X_test=None, y_test=None):
        minibatches = self.get_minibatches(X_train, y_train, minibatch_size)
        for i in range(epochs):
            loss = 0
            w_cache = [np.zeros_like(weights) for weights in self.weights]
            b_cache = [np.zeros_like(biais) for biais in self.biais]
            if verbose:
                print("Epoch {0}".format(i + 1))
            for X_mini_batch, y_mini_batch in minibatches:
                delta_W, delta_B = self.backpropagation(X_mini_batch, y_mini_batch)
                self.adagrad_update(w_cache, b_cache, delta_W, delta_B, alpha=alpha)
            if verbose:
                train_acc = self.accuracy(X_train, y_train)
                test_acc = self.accuracy(X_test, y_test)
                train_loss = (self.cost_f).fn(self.prediction(X_train), y_train)
                test_loss = (self.cost_f).fn(self.prediction(X_test), y_test)
                print("Training Loss = {0} | Training Loss = {1} | Training Accuracy = {2} | Test Accuracy = {3}".format(train_loss, test_loss, train_acc, test_acc))

    def adam_update(self, t, w_cache, b_cache, w_velocity, b_velocity, delta_W, delta_B, alpha=1, beta1=0.9, beta2=0.999):
        for i in range(len(self.layers) - 1):
            # Mise à jour du réseau
            w_cache[i] = beta1 * w_cache[i] + (1. - beta1) * np.mean(delta_W[i], axis = 0)
            b_cache[i] = beta1 * b_cache[i] + (1. - beta1) * np.mean(delta_B[i], axis = 0)

            w_velocity[i] = beta2 * w_velocity[i] + (1. - beta2) * (np.mean(delta_W[i], axis = 0)**2)
            b_velocity[i] = beta2 * b_velocity[i] + (1. - beta2) * (np.mean(delta_B[i], axis = 0)**2)

            w_mt = w_cache[i] / (1. - beta1**(t))
            b_mt = b_cache[i] / (1. - beta1**(t))

            w_vt = w_velocity[i] / (1. - beta2**(t))
            b_vt = b_velocity[i] / (1. - beta2**(t))

            self.weights[i] -= alpha * w_mt / (np.sqrt(w_vt) + 1e-8)
            self.biais[i] -= alpha * b_mt / (np.sqrt(b_vt) + 1e-8)

    def adam(self, X_train, y_train, minibatch_size, epochs, alpha, verbose=True, X_test=None, y_test=None):

        beta1 = 0.9
        beta2 = 0.999
        minibatches = self.get_minibatches(X_train, y_train, minibatch_size)
        for i in range(epochs):
            loss = 0
            w_velocity = [np.zeros_like(weights) for weights in self.weights]
            b_velocity = [np.zeros_like(biais) for biais in self.biais]
            w_cache = [np.zeros_like(weights) for weights in self.weights]
            b_cache = [np.zeros_like(biais) for biais in self.biais]
            if verbose:
                print("Epoch {0}".format(i + 1))
            t = 1
            for X_mini_batch, y_mini_batch in minibatches:
                delta_W, delta_B = self.backpropagation(X_mini_batch, y_mini_batch)
                self.adam_update(t, w_cache, b_cache, w_velocity, b_velocity, delta_W, delta_B, alpha, beta1, beta2)
                t += 1
            if verbose:
                train_acc = self.accuracy(X_train, y_train)
                test_acc = self.accuracy(X_test, y_test)
                train_loss = (self.cost_f).fn(self.prediction(X_train), y_train)
                test_loss = (self.cost_f).fn(self.prediction(X_test), y_test)
                print("Training Loss = {0} | Training Loss = {1} | Training Accuracy = {2} | Test Accuracy = {3}".format(train_loss, test_loss, train_acc, test_acc))
