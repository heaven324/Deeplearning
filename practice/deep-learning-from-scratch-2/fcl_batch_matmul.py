# 완전연결 계층 배치에 의한 행렬곱
import numpy as np
W1 = np.random.randn(2,4)
b1 = np.random.randn(4)
x = np.random.randn(10,2) # 배치 : 10
h = np.matmul(x, W1) + b1
print(h)