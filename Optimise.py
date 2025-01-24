import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

from sklearn.preprocessing import StandardScaler
from Vectorize import Vectorize

class Optimise:

    @staticmethod
    def apply_truncated_svd(feature_matrix, n_components = 30):
        """
        Apply Truncated Singular Value Decomposition (TruncatedSVD) to reduce dimensionality of the feature matrix.

        :param feature_matrix: scipy.sparse matrix or dense numpy array of features
        :param n_components: Number of components to retain
        :return: Reduced feature matrix, TruncatedSVD model
        """
        print(f"Applying TruncatedSVD to reduce dimensions to {n_components}...")
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced_features = svd.fit_transform(feature_matrix)

        print("TruncatedSVD complete. Explained variance ratio:")
        print(svd.explained_variance_ratio_)
        print(f"Total explained variance: {svd.explained_variance_ratio_.sum():.2f}")

        return reduced_features, svd

    @staticmethod
    def apply_pca(feature_matrix, n_components = 30):
        pca = PCA(n_components=30)
        reduced_features = pca.fit_transform(feature_matrix)
        print("PCA complete. Explained variance ratio:")
        print(pca.explained_variance_ratio_)
        print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.2f}")

        return reduced_features, pca


