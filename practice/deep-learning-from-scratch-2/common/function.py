# coding: utf-8
# =============================================================================
## layers function
#  2. cross entropy error = -(1/N)*sum(n, sum(k, t_nk*log y_nk))
# =============================================================================


from common.np import *


# function define
def sigmoid(x):
    """
    sigmoid function   1(page 72)
        - 신경망의 표현력을 높이는 활성화 함수, 비선형 함수
          알파벳 's'자 모형의 0 ~ 1 결과를 출력해주는 함수
	
    1. x : 임의의 실수 또는 array의 형태
	
    이론 : 1). sigmoid = 1 / ( 1 + exp(-x) )
    """
    return 1 / (1 + np.exp(-x))

def relu(x):
    """
    Relu function   1(page 77)
        - 신경망의 표현력을 높이는 활성화 함수, 비선형 함수
          0보다 작은 수는 0으로 출력하고, 0보다 큰 수는 그대로 출력해주는 함수

    1. x : 임의의 실수 또는 array의 형태

    이론 : 1). Relu = 0 or x
    """
    return np.maximum(0, x)

def softmax(x):
    """
    Softmax function   1(page 94)
        - 입력값을 정규화(출력의 합이 1)해서 출력하는 함수, 보통 마지막 출력층에서 
          사용

    1. x : 정규화 되기 전의 입력값(array의 형태)

    이론 : 1). Softmax = exp( s_k - c ) / ∑{i=1, n}( exp(s_i - c) )  # c = max(x)

    설명 : x가 2차원인 경우 행을 기준으로 비교를 해야하기 때문에 행 별로 max값을 추출
           (max값을 추출해 x에 빼주는 이유는 inf라는 불안정한 값이 나오지 않게 하기
            위해서, 결과에는 차이가 없다.) 한 후 이론에 대입
           x가 1차원인 경우 max값을 추출한 후 이론에 대입

    """
    if x.ndim == 2:
        x = x - x.max(axis = 1, keepdims = True)
        x = np.exp(x)
        x /=x.sum(axis = 1, keepdims = True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))
    return x

def cross_entropy_error(y, t):
    """
    Cross Entropy Error {for mini batch}   1(page 115), 2(page 41)
        - 신경망의 정답 추측과 정답 레이블과의 손실을 구해주는 함수(적을수록 좋다)
          참고로 정답이 아닌 다른 라벨의 확률은 t의 원핫 라벨을 곱하기 때문에
          무시된다

    1. y : 신경망의 최종 추측

    2. t : 정해져있는 정답 라벨

    이론 : 1). CEE = -∑ {k} (t_k * log y_k)
           2). CEE {for batch} = - [ ∑ {n} ∑ {k} (t_nk * log y_nk) ] / N

    설명 : 만약 배치처리가 아니라면(1차원 array의 형태) 2차원 배열로 차원을 늘린다.
           그 후 y와 t의 사이즈가 같을 때, 정답 레이블의 index를 얻어
           (y애서 정답의 위치를 찾기 위해서) 이론에 대입한다.
    """
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    if t.size == y.size:
        t = t.argmax(axis = 1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
