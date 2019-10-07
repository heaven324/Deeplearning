import numpy as np
# =============================================================================
# 100 numpy exercises사이트에서 넘파이 연습 가능
# https://github.com/rougier/numpy-100
# =============================================================================


# 정보 표시
x = np.array([1,2,3])
x.__class__
print(x.shape)
print(x.ndim)

W = np.array([[1,2,3], [4,5,6]])
print(W.shape)
print(W.ndim)


# 행렬의 원소별 연산
W = np.array([[1,2,3], [4,5,6]])
X = np.array([[0,1,2], [3,4,5]])
print(W + X)
print(W * X)


# 브로드 캐스트
A = np.array([[1,2], [3,4]])
print(A*10)  # 스칼라 브로드 캐스트

A = np.array([[1,2], [3,4]])
b = np.array([10, 20])
print(A * b)

# 벡터의 내적
a = np.array([1,2,3])
b = np.array([4,5,6])
print(np.dot(a, b))

# 행렬의 곱
A = np.array([[1,2], [3,4]])
B = np.array([[5,6], [7,8]])
print(np.matmul(A, B))