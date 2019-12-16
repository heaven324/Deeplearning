import numpy as np 
import matplotlib.pyplot as plt 
import sequence
from common.optimizer import Adam
from trainer import Trainer 
from common.util import eval_seq2seq
from seq2seq import Seq2seq 
# from peeky_seq2seq import PeekySeq2seq

# read dataset
(x_train, t_train), (x_test, t_test) = sequence.load_data('addition.txt')
char_to_id, id_to_char = sequence.get_vocab()

# set input reversal
is_reverse = False
if is_reverse:
	x_train, x_test = x_train[:, ::-1], x_test[:, ::-1]

# hyperparameter setting
vocab_size = len(char_to_id)
wordvec_size = 16
hidden_size = 128
batch_size = 128
max_epoch = 25
max_grad = 5.0

# normal or peeky setting
model = Seq2seq(vocab_size, wordvec_size, hidden_size)
# model = PeekySeq2seq(vocab_size, wordvec_size, hidden_size)

optimizer = Adam()
trainer = Trainer(model, optimizer)

acc_list = []
for epoch in range(max_epoch):
	trainer.fit(x_train, t_train, max_epoch = 1, batch_size = batch_size,
		        max_grad = max_grad)

	correct_num = 0
	for i in range(len(x_test)):
		question, correct = x_test[[i]], t_test[[i]]
		verbose = i < 10
		correct_num += eval_seq2seq(model, question, correct, 
			                        id_to_char, verbose, is_reverse)

	acc = float(correct_num) / len(x_test)
	acc_list.append(acc)
	print('Verification Accuracy %.3f%%'%(acc * 100))

# plot
x = np.arange(len(acc_list))
plt.plot(x, acc_list, marker = 'o')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.ylim(0, 1.0)
plt.show()
