from sklearn.linear_model import LogisticRegression
from Encrypt import CKKS_Encryptor
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import time
import numpy as np

class LogisticRegressionController:
    def __init__(self):
        self.model = LogisticRegression(max_iter= 1000, solver ='lbfgs')
        self.encryptor = CKKS_Encryptor()
        self.encryption_context = self.encryptor.get_encryption_context()
        self.weights = None
        self.intercept = None


    def train(self, X, y):
        self.model.fit(X, y)
        self.weights = self.model.coef_.flatten().tolist()
        self.intercept = float(self.model.intercept_[0])
        print("Logistic Regression model trained.")
        print()
        print()


    



    def makeInference(self, X_test_encrypted):
        """
        Compute encrypted dot products using encrypted FVMs and plaintext weights.
        Returns encrypted logits (before sigmoid).
        """
        start = time.time()
        encrypted_logits = []
        for enc_row in X_test_encrypted:
            dot = (enc_row * self.weights).sum() + self.intercept
            encrypted_logits.append(dot)
        end = time.time()
        print(f"Time taken for encrypted inference: {end - start:.4f} seconds")
        return encrypted_logits


    def makeInference_plain(self, X_test):
        """
        Compute plaintext logits for Logistic Regression:
          z = X·w + b
        Returns a 1D numpy array of raw scores.
        """
        # ensure weight vector is numpy array
        W = np.array(self.weights)           # shape (n_features,)
        # X_test can be dense ndarray or sparse – convert if needed
        if hasattr(X_test, "toarray"):
            X = X_test.toarray()
        else:
            X = X_test
        # compute logits
        logits = X @ W + self.intercept      # shape (n_samples,)
        return logits


