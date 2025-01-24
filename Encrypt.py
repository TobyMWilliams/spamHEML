import numpy as np
from tqdm import tqdm
import random
import time
import secrets
from multiprocessing import Pool, cpu_count
from sympy import randprime
from math import gcd

class PaillierEncryptor:

    def __init__(self, bit_length=1024):
        """
        Initialize the encryptor by generating a key pair.
        Args:
            bit_length (int): Bit length for the keys. Default is 1024.
        """
        # Generate keys
        self.p, self.q = self._generate_prime(bit_length // 2), self._generate_prime(bit_length // 2)
        self.n = self.p * self.q
        self.g = self.n + 1 
        self.n_square = self.n * self.n

    def _generate_prime(self, bit_length):
        """Generate a random prime number using sympy library. 
        Args:
            bitlength(int): bit length of prime generated.
        Returns:
            int: prime num of specified length.
        """
        lower_bound = 2**(bit_length - 1)
        upper_bound = 2**bit_length - 1
        return randprime(lower_bound, upper_bound)



    def _encrypt_value(self, m):
        """Encrypt a single value using the Paillier encryption scheme."""
        r = secrets.randbelow(self.n - 1) + 1  
        g_m = pow(self.g, m, self.n_square)
        r_n = pow(r, self.n, self.n_square)
        c = (g_m * r_n) % self.n_square
        return c
    

    def _encrypt_row(self, row):
        """Encrypt a single row."""
        return [self._encrypt_value(int(value)) for value in row]
    

    def _l_function(self, x, n):
        """L function for Paillier decryption."""
        return (x - 1) // n

    def encrypt_feature_matrix(self, matrix):
        """
        Encrypt a matrix using Paillier encryption with GPU acceleration.
        Args:
            matrix (np.ndarray): The plaintext feature vector matrix.
        Returns:
            tf.Tensor: Encrypted feature vector matrix.
        """

        start_train = time.time()

        if hasattr(matrix, "toarray"):  
            print("matrix started as sparse representation")
            matrix = matrix.toarray()
        
        encrypted_matrix = []
        
        print("Encrypting feature matrix...")
        
        with Pool(processes=cpu_count()) as pool:
            #  tqdm to show the progress bar of encryption
            encrypted_matrix = list(
                tqdm(pool.imap(self._encrypt_row, matrix), total=len(matrix), desc="Encrypting rows")
            )
        
        encrypted_matrix = np.array(encrypted_matrix, dtype=object)
        encrypted_matrix = encrypted_matrix % (2**64 - 1) 
        
        # normalising the values retains their meaning whilst preventing overflow
        max_value = np.max(encrypted_matrix)
        min_value = np.min(encrypted_matrix)
        encrypted_matrix = (encrypted_matrix - min_value) / (max_value - min_value)

        train_time = time.time() - start_train
        formatted_time = f"{train_time:.2f} seconds"

        print(f"{'Time taken to encrypt FVM:':<35}{formatted_time}")


        return encrypted_matrix

