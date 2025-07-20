import numpy as np
from tqdm import tqdm
import tenseal as ts 
import time
from multiprocessing import Pool, cpu_count




# class to house all the encryption logic 
class CKKS_Encryptor:
    def __init__(self):
        self.context = ts.context(
            scheme=ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60]
        )
        self.context.global_scale = 2 ** 40
        self.context.generate_galois_keys()
        self.context.generate_relin_keys()



    def get_encryption_context(self):

        return self.context

    def set_encryption_context(self, context):
    
        self.context = context

    def encrypt_value(self, value):
        
        return ts.ckks_vector(self.context, [float(value)])

    def encrypt_row(self, row):
       
        return ts.ckks_vector(self.context, list(map(float, row)))

    def encrypt_feature_matrix(self, matrix):
    
        start_time = time.time()
        if hasattr(matrix, "toarray"):  # Convert sparse representation to dense if necessary
            print("Converting sparse matrix to dense representation...")
            matrix = matrix.toarray()

        #print("Encrypting feature matrix with packing...")
        encrypted_matrix = [self.encrypt_row(row) for row in tqdm(matrix, desc="Encrypting rows")]
        elapsed = time.time() - start_time
        print(f"Time taken to encrypt feature matrix: {elapsed:.2f} seconds")
        return np.array(encrypted_matrix, dtype=object)

    def encrypt_model_weights(self, model):
        
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



