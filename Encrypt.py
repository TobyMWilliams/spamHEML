import numpy as np
from tqdm import tqdm
import tenseal as ts 
import time
from multiprocessing import Pool, cpu_count

class CKKS_Encryptor:
    def __init__(self):
        """
        Initialize the encryptor using CKKS scheme.
        """
        self.context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()


    def get_encryption_context(self):
        """
        Return the CKKS encryption context for use in other modules.
        """
        return self.context

    def set_encryption_context(self, context):
        """
        Set a new encryption context.
        """
        self.context = context

    def encrypt_value(self, value):
        """
        Encrypt a single float value using CKKS encryption.
        """
        return ts.ckks_vector(self.context, [float(value)])

    def encrypt_row(self, row):
        """
        Encrypt a single row by packing the entire row into one ciphertext.
        """
        return ts.ckks_vector(self.context, list(map(float, row)))

    def encrypt_feature_matrix(self, matrix):
        """
        Encrypt a matrix using CKKS encryption with row packing.
        """
        start_time = time.time()
        if hasattr(matrix, "toarray"):  # Convert sparse representation to dense if necessary
            print("Converting sparse matrix to dense representation...")
            matrix = matrix.toarray()

        print("Encrypting feature matrix with packing...")
        encrypted_matrix = [self.encrypt_row(row) for row in tqdm(matrix, desc="Encrypting rows")]
        elapsed = time.time() - start_time
        print(f"Time taken to encrypt feature matrix: {elapsed:.2f} seconds")
        return np.array(encrypted_matrix, dtype=object)

    def encrypt_model_weights(self, model):
        """
        Encrypt the model's weights by packing them into a single CKKS vector.
        """
        if not hasattr(model, "coef_") or model.coef_ is None:
            raise ValueError("Model has not been trained. No coefficients found.")

        # Flatten weights and pack into one vector
        weights = model.coef_.flatten()
        intercept = float(model.intercept_[0])
        print(f"Encrypting {len(weights)} weights into a single vector...")
        encrypted_weights = ts.ckks_vector(self.context, list(map(float, weights)))
        # For the intercept, we create a one-element vector
        encrypted_intercept = ts.ckks_vector(self.context, [intercept])
        print("Model weights encrypted successfully.")
        return encrypted_weights, encrypted_intercept

    # Helper serialization functions remain similar
    def serialize_encrypted_matrix(self, encrypted_matrix):
        return [vec.serialize() for vec in encrypted_matrix]

    def deserialize_encrypted_matrix(self, serialized_matrix):
        return [ts.ckks_vector_from(self.context, vec_bytes) for vec_bytes in serialized_matrix]

    def serialize_context(self):
        return self.context.serialize()

    def deserialize_context(self, serialized_context):
        return ts.context_from(serialized_context)







    # #older version
    # def get_encryption_context(self):
    #     """
    #     Return the CKKS encryption context for use in other modules.
    #     """
    #     return self.context
    



    # def encrypt_value(self, value):
    #     """
    #     Encrypt a single float value using CKKS encryption.
    #     """
    #     return ts.ckks_vector(self.context, [value])




    # def encrypt_vector(self, vector):
    #     """Encrypt an entire list of floats as a single CKKSVector."""
    #     return ts.ckks_vector(self.context, [float(v) for v in vector])

    # def encrypt_row(self, row):
    #     return self.encrypt_vector(row)



    # def encrypt_feature_matrix(self, matrix):
    #     start_time = time.time()
    #     if hasattr(matrix, "toarray"):
    #         print("Matrix started as sparse representation")
    #         matrix = matrix.toarray()

    #     print("Encrypting feature matrix...")
    #     # Each row is now encrypted as a single CKKSVector
    #     encrypted_matrix = [self.encrypt_row(row) for row in tqdm(matrix, desc="Encrypting rows")]
    #     elapsed = time.time() - start_time
    #     print(f"Time taken to encrypt FVM: {elapsed:.2f} seconds")
    #     return encrypted_matrix

    # def encrypt_model_weights(self):
    #     weights = self.model.coef_.flatten().tolist()
    #     intercept = float(self.model.intercept_[0])
    #     encrypted_w = self.encryptor.encrypt_vector(weights)
    #     encrypted_b = self.encryptor.encrypt_value(intercept)
    #     return encrypted_w, encrypted_b
