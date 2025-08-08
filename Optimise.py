import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA, SparsePCA



from sklearn.preprocessing import StandardScaler
from Vectorize import TFIDFVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class Optimise:


    # Sparse Dimensional reduction 
    @staticmethod
    def apply_sparse_pca(feature_matrix, k):
        # print("applying SparsePCA.")

        spca = SparsePCA(k=k, random_state=42)
        reduced_features = spca.fit_transform(feature_matrix)
        print(f"pca reduction to {k} complete.")
        return reduced_features, spca
        


    #function to apply NMF to the feature matrix
    @staticmethod
    def apply_nmf(feature_matrix, k):
        if hasattr(feature_matrix, 'toarray') and not isinstance(feature_matrix, np.ndarray):
            feature_matrix = feature_matrix.toarray()
        nmf = NMF(
            k=k,
            init='nndsvda',
            solver='mu',
            beta_loss='frobenius',
            random_state=42,
            max_iter=1000
        )        
        reduced_features = nmf.fit_transform(feature_matrix)
        print(f"nmf reduction to {k} complete.")
        return reduced_features, nmf


    #function to apply truncated SVD to the feature matrix
    @staticmethod
    def apply_truncated_svd(feature_matrix, k ):
       

        svd = TruncatedSVD(k, random_state=42)
        reduced_features = svd.fit_transform(feature_matrix)
        print(f"svd reduction to {k} complete.")
        return reduced_features, svd




                # Dense Dimensional Reduction. 

    #function to apply ICA to the feature matrix
    @staticmethod
    def apply_ica(feature_matrix, k):
        
        ica = FastICA(k=k, random_state=42, max_iter=1000)
        reduced_features = ica.fit_transform(feature_matrix)
        print(f"ica reduction to {k} complete.")
        return reduced_features, ica


    #function to apply PCA to the feature matrix
    @staticmethod
    def apply_pca(feature_matrix, k):

        pca = PCA(k)
        reduced_features = pca.fit_transform(feature_matrix)
        print(f"pca reduction to {k} complete.")
        return reduced_features, pca
    
    #function to apply Chi reduction to the feature matrix
    @staticmethod
    def apply_chi2(feature_matrix, labels, k):
        
        print("applying Chi2 feature selection.")
        if not isinstance(feature_matrix, np.ndarray) and hasattr(feature_matrix, "tocsr"):
            feature_matrix = feature_matrix.tocsr()

        chi = SelectKBest(score_func=chi2, k=k)
        reduced_features = chi.fit_transform(feature_matrix, labels)
        print(f"chi2 reduction to {k} complete.")
        return reduced_features, chi




