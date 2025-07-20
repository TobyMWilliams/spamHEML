import sys
import os
import pickle
import numpy as np
from sklearn.preprocessing import MinMaxScaler 
from tqdm import tqdm
import time


from Optimise import Optimise
from Read import Read

def save_matrix(matrix, filepath):
    """Save the matrix to a file."""
    with open(filepath, 'wb') as f:
        pickle.dump(matrix, f)


def load_matrix(filepath):
    """Load the matrix from a file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)
    

    
def to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else X


#though not scalable, this works for now to extract trends
def scale(train, test):

    all = np.vstack([train, test])
    scaler = MinMaxScaler(feature_range=(1, 4))
    all_scaled = scaler.fit_transform(all)
    train = all_scaled[:len(train)]
    test = all_scaled[:len(test)]
    return train, test


def global_scale(train, test):
    #attempting to preserve non-negativity and vector directions
    all_data = np.vstack([train, test])
    scaler = MinMaxScaler(feature_range=(0, 1))

    all_scaled = scaler.fit_transform(all_data)

    train_scaled = all_scaled[:len(train)]
    test_scaled = all_scaled[len(train):]
    return train_scaled, test_scaled

def to_dense(X):
    return X.toarray() if hasattr(X, 'toarray') else X






def main():


    training_file = "data/train.jsonl"
    test_file     = "data/test.jsonl"

    # train_data_texts  = Read.read_jsonl_text(training_file)
    labels = Read.read_jsonl_label(training_file)


    
    in_paths = {
        #sparse data
        "bow_train": "data/featureData/CHIreduced/bow_chi_train.pkl",
        "bow_test": "data/featureData/CHIreduced/bow_chi_test.pkl",

        "tfidf_train": "data/featureData/CHIreduced/tfidf_chi_train.pkl",
        "tfidf_test": "data/featureData/CHIreduced/tfidf_chi_test.pkl",

        "trigram_train": "data/featureData/CHIreduced/trigram_chi_train.pkl",
        "trigram_test": "data/featureData/CHIreduced/trigram_chi_test.pkl",

        "bigram_train": "data/featureData/CHIreduced/bigram_chi_train.pkl",
        "bigram_test": "data/featureData/CHIreduced/bigram_chi_test.pkl",

        # dense data
        "glove_train": "data/featureData/CHIreduced/GloVe_chi_train.pkl",
        "glove_test": "data/featureData/CHIreduced/GloVe_chi_test.pkl",
    
        
    }
    

    base = "data/featureData"
    noFeatures = {
        # "30": "30Features",
        "45": "45Features",
        "60": "60Features",
    }
    feature = {
        # "bow"    : "BoW",
        # "bigram" : "BIGRAM",
        # "trigram": "TRIGRAM",
        # "glove"  : "GloVe",
        "tfidf"  : "TFIDF",

    }
    

    for key, name in noFeatures.items():
        # set no. components in line with the directory
        if (key == "30"):
            components = 30
        if (key == "45"):
            components = 45
        if (key == "60"):
            components = 60
        # extend the directory
        baseplus = os.path.join(base, name)
        for key, name in feature.items():

            # this is now "data/featureData/30Features/TFIDF"
            direct = os.path.join(baseplus, name)
            print(f"chi reduced {name} train shape")
            train = load_matrix(in_paths[f"{key}_train"])
            test = load_matrix(in_paths[f"{key}_test"])

            print("\n")
            print(f"\n{name}")
            # print (f"chi reduced train shape = {train.shape}")
            # print (f"chi reduced test shape  = {test.shape}")
            
            
            print("\n")

            try:
                print(f"\n[Chi2] {name} {components}")
                path2 = os.path.join(direct, "chi2_")
                path21 = (F"{path2}train.pkl")
                path22 = (F"{path2}test.pkl")

                start_train4 = time.time()
                Train2, model = Optimise.apply_chi2(train, labels, components)
                train_time4 = time.time() - start_train4
                print(f"\n[Chi2] {name} {components} train feature extraction:  {train_time4:.5f} seconds")
                start_train44 = time.time()
                Test2 = model.transform(test)
                train_time44 = time.time() - start_train44
                print(f"\n[Chi2] {name} {components} test feature transformation:  {train_time44:.5f} seconds")

                save_matrix(Train2, path21)
                save_matrix(Test2, path22)
                
            except Exception as e:
                print(f"[!] [Chi2] {name} {components} failed: {e}")
   

               

            


            try:
        
                print(f"\n[SVD] {name} {components}")
                path2 = os.path.join(direct, "svd_")
                path21 = (F"{path2}train.pkl")
                path22 = (F"{path2}test.pkl")
                
                start_train1 = time.time()
                Train1, model = Optimise.apply_truncated_svd(train, components)
                train_time1 = time.time() - start_train1
                print(f"\n[SVD] {name} {components} train feature extraction:  {train_time1:.5f} seconds")
                

                start_train11 = time.time()
                Test1 = model.transform(test)
                train_time11 = time.time() - start_train11


                print(f"\ntrain shape = {Train1.shape} ")
                print(f"\ntest shape = {Test1.shape} ")
                print(f"\n[SVD] {name} {components} test feature transformation:  {train_time11:.5f} seconds")

                save_matrix(Train1, path21)
                save_matrix(Test1, path22)
            except Exception as e:
                print(f"[!] [SVD] {name} {components} failed: {e}")


            
           



            try:
                print(f"\n[PCA] {name} {components}")
                path4 = os.path.join(direct, "pca_")
                path41 = (F"{path4}train.pkl")
                path42 = (F"{path4}test.pkl")
                start_train2 = time.time()
                train1 = to_dense(train)
                Train4, model = Optimise.apply_pca(train1, components)
                train_time2 = time.time() - start_train2
                print(f"\n[PCA] {name} {components} train feature extraction:  {train_time2:.5f} seconds")


                start_train22 = time.time()
                test1 = to_dense(test)
                Test4 = model.transform(test1)
                train_time22 = time.time() - start_train22

                print(f"\n[PCA] {name} {components} test feature transformation:  {train_time22:.5f} seconds")
                print(f"\ntrain shape = {Train4.shape} ")
                print(f"\ntest shape = {Test4.shape} ")

                save_matrix(Train4, path41)
                save_matrix(Test4, path42)
            except Exception as e:
                print(f"[!]\n[PCA] {name} {components} failed: {e}") 



            try:
                print(f"\n[ICA] {name} {components}")
                path5 = os.path.join(direct, "ica_")
                path51 = (F"{path5}train.pkl")
                path52 = (F"{path5}test.pkl")
                

                start_train3 = time.time()
                train2 = to_dense(train)
                Train5, model = Optimise.apply_ica(train2, components)
                train_time3 = time.time() - start_train3
                print(f"\n[ICA] {name} {components} train feature extraction:  {train_time3:.5f} seconds")



                start_train33 = time.time()
                test2 = to_dense(test)
                Test5 = model.transform(test2)

                train_time33 = time.time() - start_train33
                print(f"\n[ICA] {name} {components} test feature transformation:  {train_time33:.5f} seconds")


                print(f"\ntrain shape = {Train5.shape} ")
                print(f"\ntest shape = {Test5.shape} ")

                save_matrix(Train5, path51)
                save_matrix(Test5, path52)
            except Exception as e:
                print(f"[!]\n[ICA] {name} {components} failed: {e}")

           

    # # Chi2 (requires labels)
        
           

            # below this point, dense input is required
            try:
                print(f"\n[NMF] {name} {components}")
                path3 = os.path.join(direct, "nmf_")
                path31 = (F"{path3}train.pkl")
                path32 = (F"{path3}test.pkl")

                start_train5 = time.time()
                train = to_dense(train)
                Train3, model = Optimise.apply_nmf(train, components)
                train_time5 = time.time() - start_train5
                print(f"\n[NMF] {name} {components} train feature extraction:  {train_time5:.5f} seconds")

                start_train55 = time.time()
                test = to_dense(test)
                Test3 = model.transform(test)
                train_time55 = time.time() - start_train55
                print(f"\n[NMF] {name} {components} test feature transformation:  {train_time55:.5f} seconds")

                print(f"\ntrain shape = {Train3.shape} ")
                print(f"\ntest shape = {Test3.shape} ")

                save_matrix(Train3, path31)
                save_matrix(Test3, path32)
            except Exception as e:
                print(f"[!] [NMF] {name} {components} failed: {e}")


            print("____________________________________________________________")
            print() 

if __name__ == "__main__":
    main()
