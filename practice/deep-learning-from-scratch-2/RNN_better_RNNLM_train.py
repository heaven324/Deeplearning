from common import config
# for GPU
# config.GPU = True
from common.optimizer import SGD
from trainer import RnnlmTrainer
from common.util import eval_perplexity, to_gpu
import ptb
from RNN_better_RNNLM import BetterRnnlm


# hyperparameter setting
batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5


# read data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_val, _, _ = ptb.load_data('val')
corpus_test, _, _ = ptb.load_data('test')

if config.GPU:
	corpus = to_gpu(corpus)
	corpus_val = to_gpu(corpus_val)
	corpus_test = to_gpu(corpus_test)

vocab_size = len(word_to_id)
xs = corpus[: -1]
ts = corpus[1 :]

# create model
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size, dropout)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# train!
best_ppl = float('inf')
for epoch in range(max_epoch):
	trainer.fit(xs, ts, max_epoch = 1, batch_size = batch_size, time_size = time_size,
		        max_grad = max_grad)
	model.reset_state()
	ppl = eval_perplexity(model, corpus_val)
	print('perplexity: ', ppl)

	if best_ppl > ppl:
		best_ppl = ppl
		model.save_params()
	else:
		lr /= 4.0
		optimizer.lr = lr

	model.reset_state()
	print('-' * 50)

# evaluation for test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)
