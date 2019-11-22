import numpy as np
from common.time_function_class import *

class SimpleRnnlm:
	def __init__(self, vocab_size, wordvec_size, hidden_size):
		V, D, H = vocab_size, wordvec_size, hidden_size
		rn = np.random.randn

		# weight initialize
		embed_W = (rn(V, D) / 100).astype('f')
		rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')  # Use xavier initial value
		rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')  # Use xavier initial value
		rnn_b = np.zeros(H).astype('f')
		affine_W = (rn(H, V) / np.sqrt(H)).astype('f')  # Use xavier initial value
		affine_b = np.zeros(V).astype('f')

		# create layers
		self.layers = [
			TimeEmbedding(embed_W),
			TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful = True),
			TimeAffine(affine_W, affine_b)
		]
		self.loss_layer = TimeSoftmaxWithLoss()
		self.rnn_layer = self.layers[1]

		# Store distributed representation of words in an instance variable
		self.params, self.grads = [], []
		for layer in self.layers:
			self.params += layer.params
			self.grads += layer.grads

	def forward(self, xs, ts):
		for layer in self.layers:
			xs = layer.forward(xs)
		loss = self.loss_layer.forward(xs, ts)
		return loss

	def backward(self, dout = 1):
		dout = self.loss_layer.backward(dout)
		for layer in reversed(self.layers):
			dout = layer.backward(dout)
		return dout

	def reset_state(self):
		self.rnn_layer.reset_state()

