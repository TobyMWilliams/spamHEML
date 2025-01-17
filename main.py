from Read import Read
from Vectorize import Vectorize
from ModelController import ModelController
from sklearn.linear_model import LogisticRegression

def main():

    # File paths
    training_file = "data/train.jsonl"
    test_file = "data/test.jsonl"

    # Step 1: Read data
    train_data_texts  = Read.read_jsonl_text(training_file)
    train_data_labels = Read.read_jsonl_label(training_file)

    test_data_texts  = Read.read_jsonl_text(test_file)
    test_data_labels = Read.read_jsonl_label(test_file)
    
    # Step 2: Vectorize data
    X_train, vectorize = Vectorize.vectorize_data(train_data_texts)
    X_test = vectorize.transform(test_data_texts)

    # Step 3: Train model
    model = LogisticRegression()
    model_controller = ModelController(model)
    model_controller.train(X_train, train_data_labels)

    # Step 4: Evaluate model
    model_controller.evaluatePerformance(X_test, test_data_labels)


if __name__ == "__main__":
    main()
