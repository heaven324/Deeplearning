from common.time_function_class import *
from common.base_model import BaseModel


class Rnnlm(BaseModel):
	"""
	RNN_simple_RNNLM과 이론이 비슷
	forward 함수 부분에서 predict 함수를 정의해 구분하고, RNN계층 대신에 LSTM계층을 도입한
	것이 차이점이다.(predict 를 추가한 이유는 7장 문장 생성에서 사용되기 때문이다.)
	"""
	def __init__(self, vocab_size = 10000, wordvec_size = 100, hidden_size = 100):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn

		# weight initialize
		embed_W = (rn(V, D) / 100).astype('f')
		lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
		lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
		lstm_b = np.zeros(4 * H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
		affine_b = np.zeros(V).astype('f')

		# create layers
		self.layers = [
		    TimeEmbedding(embed_W),
		    TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True),
		    TimeAffine(affine_W, affine_b)
		]
		self.loss_layer = TimeSoftmaxWithLoss()
		self.lstm_layer = self.layers[1]

		# Store distributed representation of words in an instance variable
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def predict(self, xs):
		for layer in self.layers:
			xs = layer.forward(xs)
		return xs

	def forward(self, xs, ts):
		score = self.predict(xs)
		loss = self.loss_layer.forward(score, ts)
		return loss

	def backward(self, dout = 1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def reset_state(self):
		self.lstm_layer.reset_state()