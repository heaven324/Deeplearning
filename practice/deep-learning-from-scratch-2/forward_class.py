# =============================================================================
# sigmoid function = 1 / ( 1 + exp(-x) )
# =============================================================================

import numpy as np

class Sigmoid:
    def __init__(self):
        self.params = []
        
    def forward(self, x):
        return 1 / (1 + np.exp(-x))
    
    
class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        
    def forward(self, x):
        W, b = self.params # test
        out = np.matmul(x, W) + b
        return out
    
