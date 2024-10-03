import numpy as np
from data_generation import generate_data_linearly_separable

class PocketAlgorithm:
    def __init__(self, num_samples, data_dim, data_std, num_iter = 1000):
        np.random.seed(0)
        self.num_samples = num_samples
        self.data, self.labels, self.w_spartor = generate_data_linearly_separable(
                                                    num_samples, data_dim, data_std, with_noise= 0.3)
        self.x = np.concatenate((self.data, np.ones((num_samples,1))), axis=1)
        self.w = np.random.randn(1, data_dim + 1)
        # self.tol = (tol+1)/num_samples
        self.num_iter = num_iter
        self.ls = []
        self.ws = [self.w]

    def fit(self):
        w_iter = self.w
        for _ in range(self.num_iter):
            ls , inds = self.loss()
            w_iter = self.w + self.labels[inds].T @ self.x[inds]
            

            output = np.dot(self.x, w_iter.T)
            pred_labels = np.where(output > 0, 1, -1).astype(int)
            misclassifid_ratio = np.sum(pred_labels != self.labels)/self.num_samples

            if misclassifid_ratio < ls:
                self.w = w_iter
                self.ws.append(self.w)

            self.ls.append(ls)


            

    def loss(self):
        output = np.dot(self.x, self.w.T)
        pred_labels = np.where(output > 0, 1, -1).astype(int)
        compare_pred_true =  pred_labels != self.labels
        num_misclassid = np.sum(compare_pred_true)
        indices_of_misclassid = np.where(compare_pred_true)[0]
        return float(num_misclassid/self.num_samples), indices_of_misclassid





    