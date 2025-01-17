
import time
from sklearn.linear_model import LogisticRegression as Log
from sklearn.metrics import confusion_matrix, matthews_corrcoef
from Read import Read
from Vectorize import Vectorize

class ModelController:
    
    def __init__(self, model):
        self.model = model



    def train(self, X, y):
        """Train the model."""

        # method to train the ML model, whilst returning the time taken to train the model
        start_train = time.time()

        self.model.fit(X, y)
        
        train_time = time.time() - start_train
        formatted_time = f"{train_time:.2f} seconds"

        print(f"{'Time taken to train model:':<35}{formatted_time}")




  
    

    def predict(self, X):
        """Predict using the trained model."""
        start_predict = time.time()
        
        y_pred = self.model.predict(X)

        classify_time = time.time() - start_predict

        formatted_time = f"{classify_time:.7f} seconds"
        print(f"{'Time taken to classify using model:':<35}{formatted_time}")

        return y_pred



    def evaluatePerformance(self, X, y):

        # X (array-like): Test feature set.
        # y (array-like): True labels.


        # calculate the confusion matrix
        predictions = self.predict(X)
        cm = confusion_matrix(y, predictions, labels=[1, 0])
        tn, fp, fn, tp = cm.ravel()

        # output the confusion matrix 
        print(f"\n{'Confusion Matrix:':<25}")
        print(f"{'True Positives (TP):':<25}{tp}")
        print(f"{'True Negatives (TN):':<25}{tn}")
        print(f"{'False Positives (FP):':<25}{fp}")
        print(f"{'False Negatives (FN):':<25}{fn}")
        print()

        print("other important metrics:")
        # calculate accuracy
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        print(f"\n{'Accuracy:':<25}{accuracy:.4f}")

        # calculate sensitivity
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        print(f"{'Sensitivity (Recall):':<25}{sensitivity:.4f}")

        # calculate specificity
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"{'Specificity:':<25}{specificity:.4f}")

        # calculate mcc
        mcc = matthews_corrcoef(y, predictions)
        print(f"{'MCC:':<25}{mcc:.4f}")



