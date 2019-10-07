import forward_class as fc
import numpy as np

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        
        # 가중치 편향 초기화
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)
        
        #계층 생성
        self.layers = [fc.Affine(W1, b1), fc.Sigmoid(), fc.Affine(W2, b2)]
        
        # 모든 가중치 리스트업(재조정을 쉽게 하기위해 모아놓기)
        self.params = []
        for layer in self.layers:
            self.params += layer.params # test
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
