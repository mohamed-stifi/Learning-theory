import numpy as np
from data_generation import generate_data_linearly_separable

class DeltaRuleLearningAlgorithm:
    def __init__(self, num_samples, data_dim, data_std, delta=0.01, num_iter=1000, lr = 0.001):
        np.random.seed(0)
        self.num_samples = num_samples
        self.data, self.labels, self.w_spartor = generate_data_linearly_separable(
                                                    num_samples, data_dim, data_std, with_noise= 0.3)
        self.x = np.concatenate((self.data, np.ones((num_samples, 1))), axis=1)
        self.w = np.random.randn(1, data_dim + 1)
        self.delta = delta
        self.num_iter = num_iter
        self.loss_history = []
        self.lr = lr


    def loss(self, w):
        output = np.dot(self.x, w.T)
        errors = self.labels - output
        return np.mean(errors**2)

    def gradient(self, w):
        output = np.dot(self.x, w.T)
        errors = self.labels - output
        # grad = -2 * np.dot(errors.T, self.x) / self.num_samples
        return errors

    def fit(self):
        w_iter = self.w
        for _ in range(self.num_iter):
            errors = self.gradient(w_iter)
            grad = 2 * np.dot(errors.T, self.x) / self.num_samples
            if np.linalg.norm(grad) <= self.delta:
                break
            w_iter = w_iter + self.lr*grad
            self.loss_history.append(self.loss(w_iter))
        
        self.w = w_iter
        return self.w, self.loss_history[-1]
