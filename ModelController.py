# ModelController.py
import time
from sklearn.metrics import confusion_matrix, matthews_corrcoef

from Encrypt import CKKS_Encryptor


class ModelController:



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
        print("Model coefficients:", self.model.coef_)
        print("Model intercept:", self.model.intercept_)
        # Instead of encrypting the weights, store them in plaintext.
        self.weights = self.model.coef_.flatten().tolist()
        self.intercept = float(self.model.intercept_[0])
        print("Plaintext model weights stored for homomorphic inference.")

    def encrypted_dot_product(self, enc_row):
        """
        Compute the homomorphic dot product between an encrypted row and the plaintext weight vector.
        Assumes that enc_row is a CKKSVector (encrypted input) and self.weights is a list of floats.
        """
        # multiply ckks by list of floats
        elementwise = enc_row * self.weights  
        # Sum the resulting slots to get the dot product.
        dot = elementwise.sum()
        # Add the plaintext intercept.
        return dot + self.intercept
    

   
    # def sigmoid_poly(self, z):
    #     """
    #     Polynomial approximation for a sigmoid-like activation:
    #         sigmoid(z) ≈ 0.5 + 0.19131 * z - 0.0045963 * z^3
    #     After each multiplication, we multiply by (1.0/scale) to reduce the scale.
    #     """
    #     scale = self.encryption_context.global_scale  # e.g., now 2**20
    #     # Compute z^2 and rescale
    #     z_sq = (z * z) * (1.0 / scale)
    #     # Compute z^3 using the rescaled z_sq, then rescale again
    #     z_cube = (z_sq * z) * (1.0 / scale)
    #     # Compute the polynomial approximation
    #     term1 = z * 0.19131
    #     term2 = z_cube * 0.0045963
    #     return 0.5 + term1 - term2

    
    def predict(self, X_test_encrypted):
        """
        For each encrypted input row, compute the dot product using the plaintext model,
        then decrypt the result and apply a threshold to determine the predicted label.
        """
        start_predict = time.time()
        predictions = []
        for enc_row in X_test_encrypted:
            dot = self.encrypted_dot_product(enc_row)
            # Decrypt to obtain a scalar result (simulation: in real-world, decryption is client-side)
            score = dot.decrypt()[0]  # assuming decrypt() returns a list
            # Apply a simple threshold (0) to decide the class.
            pred = 1 if score >= 0 else 0
            predictions.append(pred)
        predict_time = time.time() - start_predict
        print(f"Time taken for encrypted inference: {predict_time:.7f} seconds")
        return predictions

    def evaluatePerformance(self, X_test_encrypted, test_data_labels):
        """
        Evaluate the encrypted model's performance by comparing decrypted predictions to true labels.
        """
        predictions = self.predict(X_test_encrypted)
        cm = confusion_matrix(test_data_labels, predictions, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        print("Confusion Matrix:")
        print(f"  True Positives: {tp}")
        print(f"  True Negatives: {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(test_data_labels, predictions)
        print("Other Important Metrics:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Sensitivity (Recall): {sensitivity:.4f}")
        print(f"  Specificity: {specificity:.4f}")
        print(f"  MCC: {mcc:.4f}")


