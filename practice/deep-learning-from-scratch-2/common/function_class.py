# =============================================================================
## activate function
#  1. sigmoid function = 1 / ( 1 + exp(-x) )

## layers function
#  1. softmax function = exp(s_k) / sum(i=1, n, exp(s_i))
#  2. cross entropy error = -(1/N)*sum(n, sum(k, t_nk*log y_nk))
# =============================================================================

from common.np import *
from common.config import GPU
from common.function import softmax, cross_entropy_error


# class define
class Affine:   # 1(page 175)
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None   # backward에서 사용하기 위해 forward의 입력 데이터 값을 저장

    def forward(self, x):
        """
        클래스 변수에서 가중치 편향을 전달받아 행렬곱 연산을 해서 리턴
        backward에서 사용하기 위해 x값도 클래스 변수로 저장

        이론 : Y = XㆍWeight + bias
        """
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        """
        클래스 변수에서 현재 가중치와 편향을 전달받아 x의 역전파(dx)를 리턴하고
        기울기에 대한 클래스 변수에 가중치와 편향(W, b)의 역전파(dW, db)를 저장

        이론 : 1). dx = ∂L/∂X = (∂L/∂Y)ㆍW^T
               2). dW = ∂L/∂W = X^Tㆍ(∂L/∂Y)
               3). db = ∂L/∂B = ∂L/∂Y
        """
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx
        # ★★★★  [...]없어도 같은 결과가 나오는데 굳이 쓰는 이유는 깊은 복사를 하기 위함
        #    깊은 복사 : 실제 덮어씌울 값의 메모리 위치에 원하는 값을 복사함
        #    얕은 복사 : 덮어 씌워질 데이터의 경로가 원하는 값의 위치로 변경됨
        #                실제로 데이터가 덮어씌워 지지는 않음
    

class MatMul:   # 2(page 54)
    """
    Affine class에서 Bias변수가 없는 클래스라고 생각하면 됨
    """
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx
    

class Softmax:  # page 못찾음
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        """
        common.function에서 정의된 softmax함수의 x결과 값을 리턴
        backward에서 사용하기 위해 out을 클래스 변수로 저장
        """
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        """
        클래스 변수에서 저장된  현재 forward결과값을 전달받아 softmax역전파 값을 리턴

        이론 : 1). dx =  self.out * dout - self.out * sumdx
        역전파 식 생각해보자 ㅜㅜ 매끄럽게 해결되지는 않네
        """
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLoss:   # 1(page 179)
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None   # backward에서 사용하기 위해 forward의 데이터 값을 저장
        self.t = None   # backward에서 사용하기 위해 forward의 데이터 값을 저장
        
    def forward(self, x, t):
        """
        common.function에서 정의된 softmax함수의 x결과 값을 뽑고(y)
        common.function에서 정의된 cross_entropy_error(y, t)의 결과값을 리턴(loss)
        클래스 변수에 y와 t를 저장
        t를 저장 할 때 원핫 벡터가 아닌 argmax로 저장 (batch_size, 1)의 형태
        """
        self.t = t
        self.y = softmax(x)
        
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis = 1)
            
        loss = cross_entropy_error(self.y, self.t)
        return loss
    
    def backward(self, dout = 1):
        """
        클래스 변수에서 현재 softmax결과값(y)과 정답레이블(t)을 전달받아
        dout 에 대한 역전파를 리턴

        이론 : dx = ∂L/∂X = (∂L/∂Z) * (Y - t)
        """
        batch_size = self.t.shape[0]
        
        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx
    
    
class Sigmoid:   # 1(page 170)
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None   # backward에서 사용하기 위해 forward의 결과 데이터 값을 저장

    def forward(self, x):
        """
        입력 데이터 x에 sigmoid식을 대입한 후(out) 클래스변수에 저장하고 out을 리턴

        이론 : sigmoid = 1 / (1 + exp(-x))
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        """
        클래스 변수에서 현재 sigmoid결과값(out)을 전달받아 dout에 대한 역전파를 리턴

        이론 : dx = ∂L/∂X = (∂L/∂y) * (1 - y) * y
        """
        dx = dout * (1.0 - self.out) * self.out
        return dx
    
    
class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None   # backward에서 사용하기 위해 forward의 결과 데이터 값을 저장
        self.t = None   # backward에서 사용하기 위해 forward의 입력 데이터 값을 저장
    
    
    def forward(self, x, t):
        """
        입력 데이터 x에 sigmoid식을 대입한 후(y) 클래스변수에 저장하고
        common.function안에 cross_entropy_error(y, t) 를 리턴


        이론 : sigmoid = 1 / (1 + exp(-x))
        """
        self.t = t
        self.y = 1 / (1 + np.exp(-x))
        
        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)
        # 왜 y를 이어붙여서 loss를 출력하지? 지금 당장에는 이해가 안되네...

        return self.loss
    
    def backward(self, dout = 1):
        """
        클래스 변수에서 현재 sigmoid결과값(y)을 전달받아 dout에 대한 역전파를 리턴

        이론 : dx = ∂L/∂X = (∂L/∂Z) * (y - t)
        """
        batch_size = self.t.shape[0]
        
        dx = (self.y - self.t) * dout / batch_size
        return dx
    

class Embedding:   # 2(page 154)
    """
    맥락(원핫표현)과 가중치의 행렬곱이 특정행을 추출하는 것 뿐이기에
    사실상 행렬곱은 필요없기 때문에 특정행을 추출하는 클래스 생성(효율성)
    """
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None
        
    def forward(self, idx):
        """
        인덱스(숫자나 array의 형태)를 입력받아 W[idx]를 리턴
        
        Embedding_layer.py 참고
        """
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out
    
    def backward(self, dout):
        """
        dW에 forward에서 썼던 idx항에 dout을 더한다
        """
        dW, = self.grads
        dW[...] = 0

        # numpy
        np.add.at(dW, self.idx, dout)

        # cupy
        #np.scatter_add(dW, self.idx, dout)

        # 원래 코드
        # for i, word_id in enumerate(self.idx):
            # dW[word_id] += dout[i]
        return None