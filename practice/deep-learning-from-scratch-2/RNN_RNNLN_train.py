from common.optimizer import SGD
from trainer import RnnlmTrainer
from common.util import eval_perplexity
import ptb
from RNN_RNNLM import Rnnlm


# hyperparameter setting 
batch_size = 20
wordvec_size = 100
hidden_size  = 100
time_size = 35     # RNN을 펼치는 크기 
lr = 20.0
max_epoch = 4
max_grad = 0.25

# read data
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, _, _ = ptb.load_data('test')
vocab_size = len(word_to_id)
xs = corpus[:-1]
ts = corpus[1:]

# create model
model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
trainer = RnnlmTrainer(model, optimizer)

# training (use clip grad)
trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad, eval_interval = 20)
trainer.plot(ylim = (0, 500))

# evaluation for test data
model.reset_state()
ppl_test = eval_perplexity(model, corpus_test)
print('test perplexity: ', ppl_test)

# save parameters
model.save_params()