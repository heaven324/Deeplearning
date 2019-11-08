import numpy as np
import pickle
from trainer import Trainer
from optimizer import Adam
from word2vec_CBOW import CBOW
from util import create_contexts_target, to_cpu
import ptb


# hyperparameter setting
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 10


# read data
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)

contexts, target = create_contexts_target(corpus, window_size)


# create model
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)


# start training
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()


# save data
word_vecs = model.word_vecs
params = {}
params['word_vecs'] = word_vecs.astype(np.float16)
params['word_to_id'] = word_to_id
params['id_to_word'] = id_to_word
pkl_file = 'cbow_params.pkl'
with open(pkl_file, 'wb') as f:
    pickle.dump(params, f, -1)
