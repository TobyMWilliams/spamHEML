from sklearn.svm import SVC
from Encrypt import CKKS_Encryptor
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import time
import numpy as np


class SVMController:
    def __init__(self):
        self.model = SVC(kernel='linear')
        self.encryptor = CKKS_Encryptor()
        self.encryption_context = self.encryptor.get_encryption_context()
        self.weights = None
        self.intercept = None

    def train(self, X, y):
        self.model.fit(X, y)
        coef = self.model.coef_
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        self.weights = coef.flatten().tolist()
        self.intercept = float(self.model.intercept_[0])
        print("SVM model trained.")
        print()
        print()
        

    def makeInference(self, X_test_encrypted):
        """
        Compute encrypted dot products (logits) with the learned SVM weights.
        Returns encrypted logits for client-side classification.
        """
        encrypted_logits = []
        for enc_row in X_test_encrypted:
            dot = (enc_row * self.weights).sum() + self.intercept
            encrypted_logits.append(dot)
        return encrypted_logits



    def makeInference_plain(self, X_test):
        """
        Compute plaintext dot products (logits) with the learned SVM weights.
        Returns raw margins for classification (no sigmoid).
        """
        weights = np.array(self.weights)
        logits = X_test @ weights + self.intercept
        return logits