# ModelController.py
import time
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np

from Encrypt import CKKS_Encryptor


class SGDcontroller:



    def __init__(self, model):
        self.model = model
        self.encryptor = CKKS_Encryptor()
        self.encryption_context = self.encryptor.get_encryption_context()
        # Store plaintext model weights for FHE-friendly inference
        self.weights = None
        self.intercept = None

    def train(self, X, y):
        """Train the model on plaintext data and store its weights (plaintext) for inference."""
        start_train = time.time()
        self.model.fit(X, y)
        train_time = time.time() - start_train
        print(f"Time taken to train model: {train_time:.2f} seconds")
        # print("Model coefficients:", self.model.coef_)
        # print("Model intercept:", self.model.intercept_)
        # Instead of encrypting the weights, store them in plaintext.
        self.weights = self.model.coef_.flatten().tolist()
        coef = self.model.coef_
        if hasattr(coef, "toarray"):
            coef = coef.toarray()
        self.weights = coef.flatten().tolist()
        self.intercept = float(self.model.intercept_[0])
        print("Plaintext model weights stored for homomorphic inference.")
        print()
        print()


    def makeInference(self, X_test_encrypted):
        """
        Compute encrypted dot products using encrypted FVMs and plaintext weights.
        Returns encrypted logits (before sigmoid).
        """
        encrypted_logits = []
        for enc_row in X_test_encrypted:
            dot = (enc_row * self.weights).sum() + self.intercept
            encrypted_logits.append(dot)
        return encrypted_logits


