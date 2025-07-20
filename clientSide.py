from sklearn.metrics import confusion_matrix, matthews_corrcoef
import numpy as np
from scipy.special import expit



# messy class simulating the client-side logic of the thresholding and decryption


class ClientSide:
    def decrypt_and_classify_LR(self, encrypted_inference, threshold, testLabels):
        decrypted_logits = [logit.decrypt()[0] for logit in encrypted_inference]
        sigmoid_probs = [1 / (1 + np.exp(-z)) for z in decrypted_logits]

        preds = [1 if p >= threshold else 0 for p in sigmoid_probs]

        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    


    def classify_plain_hinge(self, logits, threshold, testLabels):
        """
        Classify based on raw SVM margins (hinge loss).
        No sigmoid; prediction is 1 if margin >= 0, else 0.
        """
        preds = [1 if p >= threshold else 0 for p in logits]

        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    


    def decrypt_and_classify_SGD(self, encrypted_inference, threshold, testLabels):
        decrypted_logits = [logit.decrypt()[0] for logit in encrypted_inference]
        sigmoid_probs = [expit(z) for z in decrypted_logits]

        preds = [1 if p >= threshold else 0 for p in sigmoid_probs]

        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm


    def decrypt_and_classify_NB(self, encrypted_inference, threshold, testLabels):
        """
        Decrypts homomorphic Naïve Bayes log-odds, thresholds at `threshold` (typically 0.0),
        and computes metrics without applying an additional sigmoid.
        """
        # Decrypt raw log-odds scores
        decrypted_scores = [logit.decrypt()[0] for logit in encrypted_inference]
        # Threshold raw scores to get predictions
        preds = [1 if z >= threshold else 0 for z in decrypted_scores]

        # Compute confusion matrix and metrics
        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    
    

    def classify_decrypted_NB(self, scores, threshold, testLabels):
        preds = [1 if z >= threshold else 0 for z in scores]
        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    




    # testing unencrypted 
    def decrypt_and_classify_LR_plain(self, encrypted_inference, threshold, testLabels):
        decrypted_logits =     encrypted_inference #[logit.decrypt()[0] for logit in encrypted_inference]
        sigmoid_probs = [1 / (1 + np.exp(-z)) for z in decrypted_logits]

        preds = [1 if p >= threshold else 0 for p in sigmoid_probs]

        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    

    def decrypt_and_classify_SGD_plain(self, encrypted_inference, threshold, testLabels):
        decrypted_logits =     encrypted_inference #[logit.decrypt()[0] for logit in encrypted_inference]
        sigmoid_probs = [expit(z) for z in decrypted_logits]

        preds = [1 if p >= threshold else 0 for p in sigmoid_probs]

        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0 
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm

    def decrypt_and_classify_NB_plain(self, encrypted_inference, threshold, testLabels):
        """
        Decrypts homomorphic Naïve Bayes log-odds, thresholds at `threshold` (typically 0.0),
        and computes metrics without applying an additional sigmoid.
        """
        # Decrypt raw log-odds scores
        decrypted_logits =     encrypted_inference #[logit.decrypt()[0] for logit in encrypted_inference]
        # Threshold raw scores to get predictions
        preds = [1 if z >= threshold else 0 for z in decrypted_logits]

        # Compute confusion matrix and metrics
        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    
    def classify_plain_LR(self, logits, threshold, testLabels):
        """
        Take raw logits (float array), apply sigmoid, then threshold.
        Returns the confusion matrix.
        """
        probs = expit(logits)                           # σ(z)=1/(1+e^{-z})
        preds = [1 if p >= threshold else 0 for p in probs]

        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        mcc = matthews_corrcoef(testLabels, preds)
        accuracy    = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) else 0
        specificity = tn / (tn + fp) if (tn + fp) else 0

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm
    

    def classify_decrypted_NB_plain(self, scores, threshold, testLabels):
        preds = [1 if z >= threshold else 0 for z in scores]
        cm = confusion_matrix(testLabels, preds, labels=[1, 0])
        tp, fn, fp, tn = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        mcc = matthews_corrcoef(testLabels, preds)

        print(f"""Confusion Matrix:
            True Positives: {tp}
            True Negatives: {tn}
            False Positives: {fp}
            False Negatives: {fn}

            Other Important Metrics:
            Accuracy: {accuracy:.4f}
            Sensitivity (Recall): {sensitivity:.4f}
            Specificity: {specificity:.4f}
            MCC: {mcc:.4f}
            """)
        return cm