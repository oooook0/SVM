import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

class LinearSVM:

    def __init__(self, D = None,K = None, alpha =0.01):
        
        reset_graph
        self.alpha = alpha
        if D and K:
            self.build(D, K)

    def build(self, D, K):
    	
        self.input = tf.placeholder(dtype=tf.float32, shape =(None, D))
        self.target = tf.placeholder(dtype=tf.float32, shape = (None, K))

        #soft margin alpha value
        alpha = tf.constant([self.alpha])

        #SVM variables
        self.A = tf.Variable(tf.random_normal(shape=[D, 1]))
        self.b = tf.Variable(tf.random_normal(shape=[K, 1]))

        def linearSVM(x):
            return tf.subtract(tf.matmul(x, self.A), self.b)

        model_output = linearSVM(self.input)

        self.predict_op = tf.sign(model_output)

        #loss term calculation
        term = tf.reduce_mean(tf.maximum(0., tf.subtract(1., tf.multiply(model_output, self.target))))
        l2_norm = tf.reduce_sum(tf.square(self.A))

        loss_op = tf.add(term, tf.multiply(alpha, l2_norm))

        return loss_op

    def fit(self, X, Y, Xtest, Ytest, batch_size = 128, learning_rate = 0.02, steps = 1500, result_freq= 75):

        #collect the matrix format
        N, D = X.shape
        N, K = Y.shape

        self.D = D
        self.K = K

        cost = self.build(self.D, self.K)

        train_op = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate).minimize(cost)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predict_op, self.target), tf.float32))

        init = tf.global_variables_initializer()

        #generate random sample from the test population
        rand_index = np.random.choice(N, size=batch_size)

        rand_x = X[rand_index]
        rand_y = Y[rand_index]

        #start the session
        with tf.Session() as session:

            session.run(init)
            last_cost = 1000000

            for i in range(steps):

                session.run(train_op, feed_dict={self.input: rand_x, self.target: rand_y})

                if i % result_freq == 0:

                    test_cost = session.run(cost, feed_dict={self.input: Xtest, self.target: Ytest})
                    Ptest = session.run(accuracy, feed_dict={self.input: Xtest, self.target: Ytest})

                    #early stopping
                    if test_cost <= last_cost: 

                        last_cost = test_cost

                        print("Cost / err at iteration j=%d: %.3f / %.3f on test set" % (i, test_cost, Ptest))

                    else:

                        break

    def predit(self,X):

        with tf.Session() as session:
            session.run(tf.global_variables_initializer())

            prediction = session.run(self.predict_op, feed_dict={self.input: X})

        return prediction

        

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

    print(x_vals_test.shape)
    print(y_vals_test.shape)

    a = LinearSVM()
    a.fit(X = x_vals_train, Y = y_vals_train, Xtest = x_vals_test, Ytest = y_vals_test, learning_rate = 0.05)
    
    print(a.predit([[5.9, 3.0, 5.1, 1.8]]))
