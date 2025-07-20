# my files
from Read import Read
from Vectorize import TFIDFVectorizer
from Encrypt import CKKS_Encryptor
from Optimise import Optimise
from SVMcontroller import SVMController
from SGDcontroller import SGDcontroller
from LogRegController import LogisticRegressionController
from GBcontroller import GBcontroller
from clientSide import ClientSide


# imported models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# imported libraries
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import os
import time
from sklearn.metrics import confusion_matrix, matthews_corrcoef, accuracy_score




def save_matrix(matrix, filepath):
    """Save the matrix to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(matrix, f)


def load_matrix(filepath):
    """Load the matrix from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    
def run_naive_bayes_inference(X_train, y_train, X_test_reduced, y_test, encryptor,  label="testNB"):
    print(f"\n=== Running Naive Bayes Inference for {label} ===")
    # encryptor = CKKS_Encryptor()

    model_controller = GBcontroller()
    model_controller.train(X_train, y_train)
    
    X_test_reduced = model_controller.scaler.transform(X_test_reduced)

    X_test_encrypted = encryptor.encrypt_feature_matrix(X_test_reduced)


    encrypted_logits = model_controller.makeInference(X_test_encrypted)
    # Decrypt ONCE
    client = ClientSide()
    # decrypted_scores = [logit.decrypt()[0] for logit in encrypted_logits]
    # print("Decrypted log-odds sample:", decrypted_scores[:5])

    client = ClientSide()
    
    cm = client.classify_decrypted_NB(
        encrypted_logits = encrypted_logits, 
        threshold = 0.0, 
        testLabels = y_test
        )

    #return cm


def run_logistic_inference(X_train, y_train, X_test_encrypted, y_test, label="testLR"):

    print(f"\n=== Running Logistic Regression Inference for {label} ===")
    
    # Step 1: Train the model
    model_controller = LogisticRegressionController()
    model_controller.train(X_train, y_train)

    # Step 2: Server-side encrypted inference (dot products)
    encrypted_logits = model_controller.makeInference(X_test_encrypted)

    # Step 3: Client-side decryption + sigmoid + thresholding
    client = ClientSide()
    cm = client.decrypt_and_classify_LR(
        encrypted_inference=encrypted_logits,
        threshold=0.553,
        testLabels=y_test
    )
    
   
    return  cm


def run_sgd_inference(X_train, y_train,    X_test_encrypted, y_test , loss,   label="testSGD",):
    print(f"\n=== Running SGDClassifier Inference for {label} ===")

    from sklearn.linear_model import SGDClassifier
    model = SGDClassifier(loss= loss, max_iter=2000, tol=1e-3, random_state=42)

    model_controller = SGDcontroller(model)
    model_controller.train(X_train, y_train)

    encrypted_logits = model_controller.makeInference(X_test_encrypted)

    client = ClientSide()

    cm = client.decrypt_and_classify_LR(  # reuse LR version (same sigmoid logic)
        encrypted_inference=encrypted_logits,
        threshold=0.558,
        testLabels=y_test
    )
    return cm


def run_svm_inference(X_train, y_train, X_test_encrypted, y_test, label="testSVM"):
    print(f"\n=== Running SVM Inference for {label} ===")

    model_controller = SVMController()
    model_controller.train(X_train, y_train)

    encrypted_logits = model_controller.makeInference(X_test_encrypted)

    client = ClientSide()
    cm = client.decrypt_and_classify_LR(  # reuse sigmoid+threshold logic
        encrypted_inference=encrypted_logits,
        threshold=0.587,
        testLabels=y_test
    )
    

    return cm



#plain-text baselines
def run_naive_bayes_inference_plain(X_train, y_train, X_test_reduced, y_test, label="testNB"):
    print(f"\n=== Running Naive Bayes Inference for {label} ===")

    model_controller = GBcontroller()
    model_controller.train(X_train, y_train)

    X_test_scaled = model_controller.scaler.transform(X_test_reduced.toarray())

    # Compute raw log-odds manually
    X_sq = np.square(X_test_scaled)
    logits = X_sq @ model_controller.w_quad + X_test_scaled @ model_controller.w_lin + model_controller.intercept

    client = ClientSide()
    cm = client.classify_decrypted_NB(
        scores=logits,
        threshold=0.0,
        testLabels=y_test
    )

    return cm


def main():




    # File paths
    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"


    
    
    # Step 1: Read data
    # train_data_texts  = Read.read_jsonl_text(training_file)
    train_data_labels = Read.read_jsonl_label(training_file)

    # test_data_texts  = Read.read_jsonl_text(test_file)
    test_data_labels = Read.read_jsonl_label(test_file)

   

    #X_train, vectorize = TFIDFVectorizer.vectorize_data(train_data_texts)




    subDir = {
        #"60" : "60Features",
        "45" : "45Features",
        #"30" : "30Features",
    }

    # this needs to be the dim red schemes
    feature = {
        "BIGRAM" : "BIGRAM",
        "TRIGRAM": "TRIGRAM",
        "TFIDF"  : "TFIDF",
        "GLOVE"  : "GloVe",
        "BOW"    : "BoW",
        }
    
    dimensionReduction ={
        # "chi2" : "chi2",
        # "pca"  : "pca",
        # "svd"  : "svd", 
        # "nmf" : "nmf",
        # "ica" : "ica",
    }
    baseDir = "data/featureData"



       

    # Step 1: Read data
    for key, name in subDir.items():

        # this is now "data/featureData/60Features"
        basePlus = os.path.join(baseDir, name)

        num = key # 60, 45, 30

        #this is now "data/featureData/matrices/60Features"
            # where the confusion matrices will be saved
        #matrixPlus = os.path.join(matrices, name)


        for key, name in feature.items():

            #this is now "data/featureData/60Features/BIGRAM"
            furtherPath = os.path.join(basePlus, name)
            feat = name 

            #this is now "data/featureData/matrices/60Features/BIGRAM"
            #furtherMatrix = os.path.join(matrixPlus, name)
            


            # iterate for each dimensional reduction type
            for key, name in dimensionReduction.items():
                
                # this is now "data/featureData/matrices/60Features/BIGRAM/chi2"
                # outMatrix = os.path.join(furtherMatrix, name)

                # this is now "data/featureData/60Features/BIGRAM/chi2"
                inData = os.path.join(furtherPath, key)
                # inData = f"data/featureData/CHIreduced/{name}_chi"
                train = load_matrix(inData + "_train.pkl")
                test  = load_matrix(inData + "_test.pkl")

                print(f"Reduced Train Matrix Shape: {train.shape}")
                print(f"Reduced Test Matrix Shape: {test.shape}")
                print()


  


                encryptor = CKKS_Encryptor()
                X_test_encrypted = encryptor.encrypt_feature_matrix(test)

                # encryption_context = encryptor.get_encryption_context()

                

                print("Encrypting reduced test data...")
                # X_test_reduced = model_controller.scaler.transform(X_test_reduced)
                # X_test_encrypted = encryptor.encrypt_feature_matrix(X_test_reduced)
                print()

                # # cm_nb = run_naive_bayes_inference(
                # #                 X_train = X_train_reduced,
                # #                 y_train = train_data_labels,
                # #                 X_test_reduced = X_test_reduced,
                # #                 y_test = test_data_labels,
                # #                 encryptor = encryptor,
                # #                 label = "test"
                # #             )
                starttime = time.time()
                cm_lr = run_logistic_inference(
                                X_train = train,
                                y_train = train_data_labels,
                                X_test_encrypted = X_test_encrypted,
                                y_test = test_data_labels,
                                label = "test"
                            )
                endtime = time.time() - starttime
                print(f"PCA {feat} {num} Logistic Regression Inference Time: {endtime:.4f} seconds")
                
                cm_svm = run_svm_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_encrypted = X_test_encrypted,
                                        y_test = test_data_labels,
                                        label = "test"
                                    )
    

    # cm_sgd = run_sgd_inference(
    #                     X_train = X_train_reduced,
    #                     y_train = train_data_labels,

    #                     X_test_encrypted = X_test_reduced,
    #                     y_test = test_data_labels,
                        
    #                     loss = "log_loss",
    #                     label = "test"
    #                 )
    # cm_sgd = run_sgd_inference(
    #                     X_train = X_train_reduced,
    #                     y_train = train_data_labels,

    #                     X_test_encrypted = X_test_reduced,
    #                     y_test = test_data_labels,

    #                     loss = "hinge",
    #                     label = "test"
    #                 )

    

    
     
    

if __name__ == "__main__":
    main()
