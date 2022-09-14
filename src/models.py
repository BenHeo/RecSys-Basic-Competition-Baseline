import numpy as np
from sklearn.decomposition import TruncatedSVD


class SVD():
    def __init__(self, sparse_matrix, truncate=100, seed=42):
        super(SVD, self).__init__()
        self.sparse_matrix = sparse_matrix
        self.truncate = truncate
        # self.model = TruncatedSVD(n_components=self.truncate, random_state=seed)
        
    def train(self):
        # self.model.fit(self.sparse_matrix)
        self.matrixs = np.linalg.svd(np.array(self.sparse_matrix), full_matrices=False)
           
    def predict(self):
        u, s, vh = self.matrixs
        truncated_u = u[:,:self.truncate]
        truncated_s = s[:self.truncate]
        truncated_vh = vh[:self.truncate,:]
        restore_matrix = np.dot(truncated_u, np.dot(np.diag(truncated_s), truncated_vh)).round().astype(int)
        
        return restore_matrix

def MF():
    
    
    return 1


def ALS():
    
    
    return 1