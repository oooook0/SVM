
import numpy as np
import random
from sklearn import datasets

class SVM():
    """
        Implementation of a Support Vector Machine using the
        Sequential Minimal Optimization (SMO) algorithm.
    """
    def __init__(self, kernels, max_iter=10000, C=1.0, epsilon=0.001):
        
        #kernels is private, need to reinit the class if to change kernel
        self.__kernels = kernels
        self._max_iter = max_iter
        self._C = C
        self._epsilon = epsilon

    @property
    def epsilon(self):
        return self._epsilon

    @epsilon.setter
    def epsilon(self, new_epsilon):
        self._epsilon = new_epsilon
    
    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, new_C):
        self._C = new_C

    @property
    def max_iter(self):
        return self._max_iter
    
    @max_iter.setter
    def max_iter(self, new_max_iter):
        self._max_iter = new_max_iter

        
    def fit(self, X, y):
        # Initialization
        n, d = X.shape
        alpha = np.zeros((n)).reshape(n,1)
        count = 0
        while True:

            count += 1
            #make a hard copy of alpha for later calculation
            alpha_prev = np.copy(alpha)

            for j in range(0, n):
                i = self.get_rnd_int(0, n-1, j) # Get random int i~=j
                x_i, x_j, y_i, y_j = X[i,:], X[j,:], y[i], y[j]
                k_ij = self.__kernels(x_i, x_i) + self.__kernels(x_j, x_j) - 2 * self.__kernels(x_i, x_j)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = alpha[j], alpha[i]
                L, H = self.compute_L_H(self._C, alpha_prime_j, alpha_prime_i, y_j, y_i)

                # Compute model parameters
                self.w = self.calc_w(alpha, y, X)
                self.b = self.calc_b(X, y, self.w)


                # Compute E_i, E_j
                E_i = self.E(x_i, y_i, self.w, self.b)
                E_j = self.E(x_j, y_j, self.w, self.b)

                # Set new alpha values
                alpha[j] = alpha_prime_j + float(y_j*(E_i - E_j))/k_ij
                alpha[j] = max(alpha[j], L)
                alpha[j] = min(alpha[j], H)
                alpha[i] = alpha_prime_i + y_i*y_j * (alpha_prime_j - alpha[j])

            # Check convergence
            diff = np.linalg.norm(alpha - alpha_prev)
            if diff < self._epsilon:
                break

            if count >= self._max_iter:
                print("Iteration number exceeded the max of %d iterations" % (self._max_iter))
                return

        # Compute final model parameters
        self.b = self.calc_b(X, y, self.w)

        if self.__kernels == Kernel.linear():
            self.w = self.calc_w(alpha, y, X)

        # Get support vectors
        alpha_idx = np.where(alpha > 0)[0]
        support_vectors = X[alpha_idx, :]
        return support_vectors, count

    @staticmethod
    def calc_b(X, y, w):
        b_tmp = y - np.dot(w.T,X.T)
        return np.mean(b_tmp)

    @staticmethod
    def calc_w(alpha, y, X):
        return np.dot(X.T, np.multiply(alpha,y))

    # Prediction
    @staticmethod
    def h(X, w, b):
        return np.sign(np.dot(w.T, X.T) + b).astype(int)

    def predict(self, X):
        return self.h(X, self.w, self.b)

    # Prediction error
    def E(self, x_k, y_k, w, b):
        return self.h(x_k, w, b) - y_k

    @staticmethod
    def compute_L_H(C, alpha_prime_j, alpha_prime_i, y_j, y_i):
        if (y_i != y_j):
            return max(0, alpha_prime_j - alpha_prime_i), min(C, C - alpha_prime_i + alpha_prime_j)
        else:
            return max(0, alpha_prime_i + alpha_prime_j - C), min(C, alpha_prime_i + alpha_prime_j)

    @staticmethod
    def get_rnd_int(a,b,z):
        i = z
        cnt=0
        while i == z and cnt<1000:
            i = random.randint(a,b)
            cnt=cnt+1
        return i

class Kernel():

    """
        Implements list of kernels from
        http://en.wikipedia.org/wiki/Support_vector_machine
    """
    # Linear kernels
    @staticmethod
    def linear():
        return lambda x1, x2: np.dot(x1, x2.T)

    # Quadratic kernels
    @staticmethod
    def quadratic():
        return lambda x1, x2: (np.dot(x1, x2.T) ** 2)

    # Radial basis function kernels
    @staticmethod
    def RBF(gamma = 10):
        return lambda x1, x2: np.exp(-gamma*np.linalg.norm(np.subtract(x1, x2)))

    # gaussian kernels
    @staticmethod
    def gaussian(sigma):
        return lambda x, y: \
            np.exp(-np.sqrt(np.linalg.norm(x-y) ** 2 / (2 * sigma ** 2)))

    @staticmethod
    def _polykernel(dimension, offset):
        return lambda x, y: (offset + np.inner(x, y)) ** dimension

    @classmethod
    def inhomogenous_polynomial(cls, dimension):
        return cls._polykernel(dimension=dimension, offset=1.0)

    @classmethod
    def homogenous_polynomial(cls, dimension):
        return cls._polykernel(dimension=dimension, offset=0.0)

    @staticmethod
    def hyperbolic_tangent(kappa, c):
        return lambda x, y: np.tanh(kappa * np.dot(x, y) + c)

if __name__ == "__main__":

    iris = datasets.load_iris()
    x_vals = np.array([[x[0],x[1],x[2], x[3]] for x in iris.data])
    y_vals = np.array([1 if y == 0 else -1 for y in iris.target])

    train_indices = np.random.choice(len(x_vals),
                                     round(len(x_vals)*0.8),
                                     replace=False)
    test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
    x_vals_train = x_vals[train_indices]
    x_vals_test = x_vals[test_indices]
    y_vals_train = np.transpose([y_vals[train_indices]])
    y_vals_test = np.transpose([y_vals[test_indices]])

    #print(x_vals_test.shape)
    #print(y_vals_test.shape)

    a = SVM(max_iter=10000, C=1.0, epsilon=0.001, kernels = Kernel.homogenous_polynomial(dimension = 2))
    suppoertV, count = a.fit(X = x_vals_train, y = y_vals_train)

    print(suppoertV)
    print(count)
    
    print((a.predict(x_vals_test) == y_vals_test.T).sum()/len(y_vals_test))
