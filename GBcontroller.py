# NBcontroller.py

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from Encrypt import CKKS_Encryptor
from sklearn.metrics import confusion_matrix, matthews_corrcoef

   
class GBcontroller:
    def __init__(self):
        self.model = GaussianNB(var_smoothing=1e-6)  # slightly higher smoothing to reduce extreme weights
        self.scaler = StandardScaler(with_mean = False)
        self.encryptor = CKKS_Encryptor()
        self.encryption_context = self.encryptor.get_encryption_context()
        self.w_quad = None
        self.w_lin = None
        self.intercept = None

    def train(self, X, y):  
        # Ensure dense input
        if not isinstance(X, np.ndarray) and hasattr(X, 'toarray'):
            X = X.toarray()

        # Fit and transform with StandardScaler
        X = self.scaler.fit_transform(X)

        # Fit the model
        self.model.fit(X, y)

        # Compute weights from trained model
        means = self.model.theta_
        vars_ = self.model.var_

        # Prevent division by very small variances
        vars_ = np.maximum(vars_, 1e-6)

        self.w_quad = -0.5 * (1.0 / vars_[1] - 1.0 / vars_[0])
        self.w_lin  = (means[1] / vars_[1]) - (means[0] / vars_[0])
        self.intercept = (
            -0.5 * (np.sum((means[1] ** 2) / vars_[1]) - np.sum((means[0] ** 2) / vars_[0]))
            - 0.5 * np.sum(np.log(vars_[1] / vars_[0]))
            + np.log(self.model.class_prior_[1] / self.model.class_prior_[0])
        )

        # Clip the weights to avoid exploding scores
        self.w_quad = np.clip(self.w_quad, -50, 50)
        self.w_lin = np.clip(self.w_lin, -50, 50)

        print("Naive Bayes (Gaussian) model trained.")
        print()
        print()

    def makeInference(self, encrypted_feature_matrix):
        encrypted_scores = []
        for enc_x in encrypted_feature_matrix:
            enc_x2 = enc_x * enc_x
            enc_q = enc_x2 * self.w_quad
            enc_l = enc_x * self.w_lin
            enc_score = enc_q.sum() + enc_l.sum() + self.intercept
            encrypted_scores.append(enc_score)

        return encrypted_scores
