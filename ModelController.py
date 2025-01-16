
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
        self.model.fit(X, y)

    def predict(self, X):
        """Predict using the trained model."""
        return self.model.predict(X)

    def evaluate(self, X, y):
        """Evaluate the model."""
        from sklearn.metrics import accuracy_score
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
