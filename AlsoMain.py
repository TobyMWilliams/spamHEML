
from Read import Read
from Vectorize import TFIDFVectorizer
from Encrypt import CKKS_Encryptor
from Optimise import Optimise
from SGDcontroller import SGDcontroller
from GBcontroller import GBcontroller
from LogRegController import LogisticRegressionController
from SVMcontroller import SVMController
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
from sklearn.metrics import confusion_matrix, matthews_corrcoef
import time



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


    start_train_svm = time.time()

    model_controller.train(X_train, y_train)
    train_time_svm = time.time() - start_train_svm

    print(f"Time taken to train GB model: {train_time_svm:.4f} seconds")



    start_test_GB = time.time()

    X_test_scaled     = model_controller.scaler.transform(X_test_reduced)
    X_test_encrypted = encryptor.encrypt_feature_matrix(X_test_scaled    )

    encrypted_logits = model_controller.makeInference(X_test_encrypted)
    test_time_GB = time.time() - start_test_GB


    # Decrypt ONCE
    client = ClientSide()
    cm = client.decrypt_and_classify_NB(
        encrypted_inference = encrypted_logits, 
        threshold = 0.0, 
        testLabels = y_test
        )
    print()
    print(f"Time taken to test enc GB model: {test_time_GB:.4f} seconds")
    print()
    #return cm







def find_optimal_threshold(logits, true_labels, apply_sigmoid=False, metric="mcc"):
   
    from sklearn.metrics import accuracy_score, f1_score

    if apply_sigmoid:
        logits = [1 / (1 + np.exp(-z)) for z in logits]

    best_thresh, best_score = 0.0, -np.inf
    for t in np.linspace(0.0, 1.0, 201): 
        preds = [1 if z >= t else 0 for z in logits]

        if metric == "mcc":
            score = matthews_corrcoef(true_labels, preds)
        elif metric == "accuracy":
            score = accuracy_score(true_labels, preds)
        elif metric == "f1":
            score = f1_score(true_labels, preds)
        else:
            raise ValueError("Unsupported metric")

        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh

def run_logistic_inference(X_train, y_train, X_test_encrypted, y_test, label):

    print(f"\n=== Running Logistic Regression Inference for {label} ===")
    start_train_log = time.time()

    # Step 1: Train the model
    model_controller = LogisticRegressionController()
    model_controller.train(X_train, y_train)

    train_time_log = time.time() - start_train_log
    print()
    print(f"Time taken to train log model: {train_time_log:.5f} seconds")
    print()
    

    start_test_log = time.time()
    # Step 2: Server-side encrypted inference (dot products)
    encrypted_logits = model_controller.makeInference(X_test_encrypted)


    # Step 3: Client-side decryption + sigmoid + thresholding
    client = ClientSide()
    cm = client.decrypt_and_classify_LR(
        encrypted_inference=encrypted_logits,
        threshold=0.553,
        testLabels=y_test
    )    
    test_time_log = time.time() - start_test_log

    print()
    print(f"Time taken to test log  model: {test_time_log:.5f} seconds")
    print()
    return  cm

def run_sgd_inference(X_train, y_train, X_test_encrypted, y_test, label, loss):
    print(f"\n=== Running SGDClassifier Inference for {label} {loss}===")

    model = SGDClassifier(loss = loss , max_iter=2000, tol=1e-3, random_state=42)

    start_train_sgd = time.time()
    model_controller = SGDcontroller(model)
    model_controller.train(X_train, y_train)
    train_time_sgd = time.time() - start_train_sgd

    print(f"Training SGDClassifier with loss: {loss}: train time: {train_time_sgd:.4f} seconds")

    start_test_SGD = time.time()
    encrypted_logits = model_controller.makeInference(X_test_encrypted)
    test_time_sgd = time.time() - start_test_SGD

    client = ClientSide()

    cm = client.decrypt_and_classify_SGD(  
        encrypted_inference=encrypted_logits,
        threshold=0.558,
        testLabels=y_test
    )  
    print()  
    print(f"Testing SGDClassifier with loss: {loss}: test time: {test_time_sgd:.4f} seconds")
    print()
    return cm

def run_svm_inference(X_train, y_train, X_test_encrypted, y_test, label):
    print(f"\n=== Running SVM Inference for {label} ===")

    model_controller = SVMController()

    start_train_svm = time.time()
    model_controller.train(X_train, y_train)
    train_time_svm = time.time() - start_train_svm
    print(f"Time taken to train SVM model: {train_time_svm:.5f} seconds")

    client = ClientSide()

    start_test_svm = time.time()
    encrypted_logits = model_controller.makeInference(X_test_encrypted)
    test_time_hinge = time.time() - start_test_svm
    print(f"Time taken to test SVM model: {test_time_hinge:.5f} seconds")

    
    #encrypted_logits = model_controller.makeInference(X_test_encrypted)
    # logits = [z.decrypt()[0] for z in encrypted_logits]
    # threshold = find_optimal_threshold(logits, y_test, apply_sigmoid=False)
    cm = client.decrypt_and_classify_LR(  # reuse sigmoid+threshold logic
        encrypted_inference=encrypted_logits,
        threshold=0.587,
        testLabels=y_test
    )
    

    return cm

def run_svm_inference_plain(X_train, y_train, X_test_reduced, y_test, label):
    print(f"\n=== Running SVM Inference (Plaintext) for {label} ===")

    model_controller = SVMController()

    start_train_hinge = time.time()
    model_controller.train(X_train, y_train)
    train_time_hinge = time.time() - start_train_hinge
    print(f"Time taken to train plain SVM model: {train_time_hinge:.5f} seconds")


    start_test_hinge = time.time()
    logits = model_controller.makeInference_plain(X_test_reduced)
    threshold = find_optimal_threshold(logits, y_test, apply_sigmoid=False)
    client = ClientSide()
    cm = client.classify_plain_hinge(
        logits=logits,
        threshold = threshold,
        testLabels = y_test)
    test_time_hinge = time.time() - start_test_hinge
    print(f"Time taken to test plain SVM model: {test_time_hinge:.5f} seconds")

    return threshold

def run_logistic_inference_plain(X_train, y_train, X_test_reduced,y_test,label,threshold=None ):
    print(f"\n=== Running Plaintext Logistic Regression for {label} ===")

    # 1) Train
    ctrl = LogisticRegressionController()
    start = time.time()
    ctrl.train(X_train, y_train)
    print(f"Training time: {time.time()-start:.2f}s\n")


    startTest = time.time()
    # 2) Compute raw logits
    logits = ctrl.makeInference_plain(X_test_reduced)
    print(f"computation time: {time.time()-startTest:.2f}s\n")

    # 3) Optionally find best threshold (uncomment if you have find_optimal_threshold)
    if threshold is None:
        threshold = find_optimal_threshold(logits, y_test, apply_sigmoid=True)
        print(f"Optimal threshold (MCC): {threshold:.3f}\n")

    # 4) Classify & print metrics
    client = ClientSide()
    cm = client.classify_plain_LR(logits, threshold=threshold, testLabels=y_test)
    print(f"Test time: {time.time()-startTest:.2f}s\n")

    return cm



def main():
    # File paths
    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"

    train_data_labels = Read.read_jsonl_label(training_file)
    test_data_labels = Read.read_jsonl_label(test_file)

    baseDir = "data/featureData"
    matrices = "data/featureData/matrices"

   
    subDir = {
        # "60" : "60Features",
        "45" : "45Features",
        # "30" : "30Features",
    }

    # this needs to be the dim red schemes
    feature = {
        # "BIGRAM" : "BIGRAM",
        # "TRIGRAM": "TRIGRAM",
        "TFIDF"  : "TFIDF",
        # "GLOVE"  : "GloVe",
        # "BOW"    : "BoW",
        }
    
    dimensionReduction ={
        # "chi2" : "chi2",
        #"pca"  : "pca",
        "svd"  : "svd", 
        #  "nmf" : "nmf",
        #  "ica" : "ica",
    }
 


       
    for key, name in dimensionReduction.items():
        drkeys = key 
    # Step 1: Read data
        for key, name in subDir.items():
            noname = name 

            # this is now "data/featureData/60Features"
            basePlus = os.path.join(baseDir, name)

            num = key # 60, 45, 30


            #this is now "data/featureData/matrices/60Features"
                # where the confusion matrices will be saved
            matrixPlus = os.path.join(matrices, name)


            for key, name in feature.items():
                featname = name

                #this is now "data/featureData/60Features/BIGRAM"

                furtherPath = os.path.join(basePlus, name)

                #this is now "data/featureData/matrices/60Features/BIGRAM"
                furtherMatrix = os.path.join(matrixPlus, name)
                name1 = name + num


            # iterate for each dimensional reduction type
          

                name2 = noname + " " + featname + " " + drkeys 
                
                # this is now "data/featureData/matrices/60Features/BIGRAM/chi2"
                # outMatrix = os.path.join(furtherMatrix, name)

                # this is now "data/featureData/60Features/BIGRAM/chi2"
                inData = os.path.join(baseDir, noname, featname,drkeys )
                # inData = f"data/featureData/CHIreduced/{name}_chi"
                train = load_matrix(inData + "_train.pkl")
                test  = load_matrix(inData + "_test.pkl")


                # encrypt test data 
                

                #        cm matrix paths
                outMatrix = "data/featureData/matrices/CHI"
                logregCM = os.path.join(outMatrix, "logregCM.pkl")
                nbCM     = os.path.join(outMatrix, "nbCM.pkl")
                lrsgdCM    = os.path.join(outMatrix, "sgdCM.pkl")
                svsgdCM    = os.path.join(outMatrix, "hinge_sgdCM.pkl")
                mhsgdCM    = os.path.join(outMatrix, "huber_sgdCM.pkl")
                svmCM    = os.path.join(outMatrix, "svmCM.pkl")


               

                cm_lr_plain = run_logistic_inference_plain(
                    train, 
                    train_data_labels,
                    test, 
                    test_data_labels, 
                    label = (f"{name2} logistic regression plain"),
                )

                cm_svm_plain = run_svm_inference_plain(
                    train, 
                    train_data_labels,
                    test, 
                    test_data_labels, 
                    label = (f"{name2} SVM plain"),
                )


                encryptor = CKKS_Encryptor()
                #encryption_context = encryptor.get_encryption_context()

                # print("Encrypting reduced test data...")
                X_test_encrypted = encryptor.encrypt_feature_matrix(test)
                print()


               
             
                cm_lr = run_logistic_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_encrypted = X_test_encrypted,
                                        y_test = test_data_labels,
                                        label = (f"{name2} logistic regression"),
                                    )
                
                cm_svm = run_svm_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_encrypted = X_test_encrypted,
                                        y_test = test_data_labels,
                                        # threshold = threshold,
                                        label = (f"{name2} svm")
                                    )
                

                cm_sgd = run_sgd_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_encrypted = X_test_encrypted,
                                        y_test = test_data_labels,
                                        label = (f"{name2} sgd(hinge)"),
                                        loss = "hinge"
                                    )
               
                cm_sgd = run_sgd_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_encrypted = X_test_encrypted,
                                        y_test = test_data_labels,
                                        label = (f"{name2} sgd(log_loss)"),
                                        loss = "log_loss"
                                    )
                
                cm_sgd = run_sgd_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_encrypted = X_test_encrypted,
                                        y_test = test_data_labels,
                                        label = (f"{name2} sgd(modified_huber)"),
                                        loss = "modified_huber"
                                    )
                cm_nb = run_naive_bayes_inference(
                                        X_train = train,
                                        y_train = train_data_labels,
                                        X_test_reduced = test,
                                        y_test = test_data_labels,
                                        encryptor = encryptor,
                                        label = (f"{name2} Gaussian Bayes"),
                                    )
                


         

               


if __name__ == "__main__":
    main()
