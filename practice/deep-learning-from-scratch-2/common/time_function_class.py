# coding: utf-8

# h_t = tanh(h_(t-1) * Wh + x_t * Wx + b)


from common.np import *
from common.function_class import *
from common.function import sigmoid


class RNN:    # 2(page 212)
    """
    RNN처리를 한단계만 수행하는 클래스 
    """
	def __init__(self, Wx, Wh, b):
		self.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
		self.cache = None

	def forward(self, x, h_prev):
		"""
		이론 : 1). RNN = tanh(h_(t-1) * Wh + x_t * Wx + b)

		설명 : 클래스 변수 params를 받아서 이론에 맞는 식에 대입한 h_next를 리턴
		       backward에서 사용하기 위해 x, h_prev, h_next를 cache에 저장
		"""
		Wx, Wh, b = self.params
		t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
		h_next = np.tanh(t)

		self.cache = (x, h_prev, h_next)
		return h_next

	def backward(self, dh_next):
		"""
		크래스변수에서 cache와 params를 받아서 역전파를 수행

		역전파 참고 : fig 5-20
		"""
		Wx, Wh, b = self.params
		x, h_prev, h_next = self.cache

		dt = dh_next * (1 - h_next ** 2)
		db = np.sum(dt, axis = 0)
		dWh = np.dot(h_prev.T, dt)
		dh_prev = np.dot(dt, Wh.T)
		dWx =np.dot(x.T, dt)
		dx = np.dot(dt, Wx.T)

		self.grads[0][...] = dWx
		self.grads[1][...] = dWh
		self.grads[2][...] = db

		return dx, dh_prev


class TimeRNN:
	def __init__(self, Wx, Wh, b, stateful = False):
		"""
		1. layers : 이 클래스 내에서 정의되는 RNN들을 모아놓은 변수
		            RNN은 forward에서 정의가 된다

		2. stateful : RNN의 은닉상태를 유지할지 말지 결정하는 스위치 변수
                      정확히는 h의 상태를 유지할지 초기화 할지 정하는 변수
		"""
		self.params = [Wx, Wh, b]
		self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
		self.layers = None

		self.h, self.dh = None, None
		self.stateful = stateful

	def set_state(self, h):
		self.h = h

	def reset_state(self):
		self.h = None

	def forward(self, xs):
		"""
        1. N : 미니 배치 크기

        2. T : 시계열 데이터의 분량

        3. D : 입력벡터의 차원수 

        forward를 여러번 호출하는 방법으로 먼저저장된 h를 유지하며 사용한다
		"""
		Wx, Wh, b = self.params
		N, T, D = xs.shape
		D,H = Wx.shape

		self.layers = []
		hs = np.empty((N, T, H), dtype = 'f')

		if not self.stateful or self.h is None:
			self.h = np.zeros((N, H), dtype = 'f')

		for t in range(T):
			layer = RNN(*self.params) # params안에 있는 3개의 변수들을 한번에 넣을 때 * 사용
			self.h = layer.forward(xs[:, t, :], self.h)
			hs[:, t, :] = self.h
			self.layers.append(layer)

		return hs

	def backward(self, dhs):
		"""
		fig 5-24 참고
		"""
		Wx, Wh, b = self.params
		N, T, H = dhs.shape
		D, H = Wx.shape

		dxs = np.empty((N, T, D), dtype = 'f')
		dh = 0
		grads = [0, 0, 0]
		for t in reversed(range(T)):
			layer = self.layers[t]
			dx, dh = layer.backward(dhs[:, t, :] + dh)
			dxs[:, t, :] = dx

			for i, grad in enumerate(layer.grads):
				grads[i] += grad

		for i, grad in enumerate(grads):
			self.grads[i][...] = grad
		self.dh = dh

		return dxs


class TimeEmbedding:
	def __init__(self, W):
		self.params = [W]
		self.grads = [np.zeros_like(W)]
		self.layers = None
		self.W = W

	def forward(self, xs):
		N, T = xs.shape
		V, D = self.W.shape

		out = np.empty((N, T, D), dtype = 'f')
		self.layers = []

		for t in range(T):
			layer = Embedding(self.W)
			out[:, t, :] = layer.forward(xs[:, t])
			self.layers.append(layer)

		return out

	def backward(self, dout):
		N, T, D = dout.shape

		grad = 0
		for t in range(T):
			layer = self.layers[t]
			layer.backward(dout[:, t, :])
			grad += layer.grads[0]

		self.grads[0][...] = grad
		return None


class TimeAffine:
	def __init__(self, W, b):
		self.params = [W, b]
		self.grads = [np.zeros_like(W), np.zeros_like(b)]
		self.x = None

	def forward(self, x):
		N, T, D = x.shape
		W, b = self.params

		rx = x.reshape(N*T, -1)
		out = np.dot(rx, W) + b
		self.x = x
		return out.reshape(N, T, -1)

	def backward(self, dout):
		x = self.x
		N, T, D = x.shape
		W, b = self.params

		dout = dout.reshape(N*T, -1)
		rx = x.reshape(N*T, -1)

		db = np.sum(dout, axis = 0)
		dW = np.dot(rx.T, dout)
		dx = np.dot(dout, W.T)
		dx = dx.reshape(*x.shape)

		self.grads[0][...] = dW
		self.grads[1][...] = db

		return dx


class TimeSoftmaxWithLoss:
	def __init__(self):
		self.params, self.grads = [], []
		self.cache = None
		self.ignore_label = -1

	def forward(self, xs, ts):
		N, T, V = xs.shape

		if ts.ndim ==3:  # 정답 vector가 one hot vector인 경우
			ts = ts.argmax(axis = 2)

		mask = (ts != self.ignore_label)

		# 배치용과 시계열용을 정리(reshape)
		xs = xs.reshape(N*T, V)
		ts = ts.reshape(N*T)
		mask = mask.reshape(N*T)

		ys = softmax(xs)
		ls = np.log(ys[np.arange(N*T), ts])
		ls *= mask   # ignore_label에 해당하는 데이터는 손실을 0으로 설정
		loss = -np.sum(ls)
		loss /= mask.sum()

		self.cache = (ts, ys, mask, (N, T, V))
		return loss

	def backward(self, dout = 1):
		ts, ys, mask, (N, T, V) = self.cache

		dx = ys
		dx[np.arange(N*T), ts] -= 1
		dx *= dout
		dx /= mask.sum()
		dx *= mask[:, np.newaxis]  # ignore_label에 해당하는 데이터는 기울기를 0으로 설정

		dx = dx.reshape((N, T, V))

		return dx

