from data_generation import generate_data_linearly_separable
import numpy as np



class PerceptronLearningAlgorithme:
    def __init__(self, num_samples, data_dim, data_std, tol=1):
        np.random.seed(0)
        self.num_samples = num_samples
        self.data, self.labels, self.w_spartor = generate_data_linearly_separable(
                                                    num_samples, data_dim, data_std
                                                    )
        self.x = np.concatenate((self.data, np.ones((num_samples,1))), axis=1)
        self.w = np.random.randn(1, data_dim + 1)
        self.tol = (tol+1)/num_samples
        self.ls = []
        self.ws = [self.w]
        
    def fit(self):
        ls , inds = self.loss()
        while ls > self.tol:
            self.ls.append(ls)
            self.w = self.w + self.labels[inds].T @ self.x[inds]
            self.ws.append(self.w)
            ls , inds = self.loss()

    def loss(self):
        output = np.dot(self.x, self.w.T)
        pred_labels = np.where(output > 0, 1, -1).astype(int)
        compare_pred_true =  pred_labels != self.labels
        num_misclassid = np.sum(compare_pred_true)
        indices_of_misclassid = np.where(compare_pred_true)[0]
        return float(num_misclassid/self.num_samples), indices_of_misclassid



