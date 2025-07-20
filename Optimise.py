import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD, NMF, FastICA, SparsePCA



from sklearn.preprocessing import StandardScaler
from Vectorize import TFIDFVectorizer
from sklearn.feature_selection import SelectKBest, chi2


class Optimise:


    # Sparse Dimensional reduction 
    @staticmethod
    def apply_sparse_pca(feature_matrix, n_components):
        # print("applying SparsePCA.")

        spca = SparsePCA(n_components=n_components, random_state=42)
        reduced_features = spca.fit_transform(feature_matrix)
        print(f"pca reduction to {n_components} complete.")
        return reduced_features, spca
        


    #function to apply NMF to the feature matrix
    @staticmethod
    def apply_nmf(feature_matrix, n_components):
        if hasattr(feature_matrix, 'toarray') and not isinstance(feature_matrix, np.ndarray):
            feature_matrix = feature_matrix.toarray()
        nmf = NMF(
            n_components=n_components,
            init='nndsvda',
            solver='mu',
            beta_loss='frobenius',
            random_state=42,
            max_iter=1000
        )        
        reduced_features = nmf.fit_transform(feature_matrix)
        print(f"nmf reduction to {n_components} complete.")
        return reduced_features, nmf


    #function to apply truncated SVD to the feature matrix
    @staticmethod
    def apply_truncated_svd(feature_matrix, n_components ):
       

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = svd.fit_transform(feature_matrix)
        print(f"svd reduction to {n_components} complete.")
        return reduced_features, svd




                # Dense Dimensional Reduction. 

    #function to apply ICA to the feature matrix
    @staticmethod
    def apply_ica(feature_matrix, n_components):
        
        ica = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        reduced_features = ica.fit_transform(feature_matrix)
        print(f"ica reduction to {n_components} complete.")
        return reduced_features, ica


    #function to apply PCA to the feature matrix
    @staticmethod
    def apply_pca(feature_matrix, n_components):

        pca = PCA(n_components)
        reduced_features = pca.fit_transform(feature_matrix)
        print(f"pca reduction to {n_components} complete.")
        return reduced_features, pca
    
    #function to apply Chi reduction to the feature matrix
    @staticmethod
    def apply_chi2(feature_matrix, labels, n_components):
        
        print("applying Chi2 feature selection.")
        if not isinstance(feature_matrix, np.ndarray) and hasattr(feature_matrix, "tocsr"):
            feature_matrix = feature_matrix.tocsr()

        chi = SelectKBest(score_func=chi2, k=n_components)
        reduced_features = chi.fit_transform(feature_matrix, labels)
        print(f"chi2 reduction to {n_components} complete.")
        return reduced_features, chi




