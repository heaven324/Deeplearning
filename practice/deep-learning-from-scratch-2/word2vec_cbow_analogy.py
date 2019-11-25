from common.util import analogy
import pickle


pkl_file = 'cbow_params.pkl'

with open(pkl_file, 'rb') as f:
    params = pickle.load(f)
    word_vecs = params['word_vecs']
    word_to_id = params['word_to_id']
    id_to_word = params['id_to_word']


# analogy process
print('-'*50)
analogy('king', 'man', 'queen', word_to_id, id_to_word, word_vecs, answer = 'woman')
# analogy('take', 'took', 'go', word_to_id, id_to_word, word_vecs)
# analogy('car', 'cars', 'child', word_to_id, id_to_word, word_vecs)
# analogy('good', 'better', 'bad', word_to_id, id_to_word, word_vecs)

# -------------------------------------------
import numpy as np

def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x


a_vec, b_vec, c_vec = word_vecs[word_to_id['take']], word_vecs[word_to_id['took']], word_vecs[word_to_id['go']]

query_vec = b_vec - a_vec + c_vec
# print(query_vec)
# print(query_vec.ndim)
# print(np.sqrt((query_vec * query_vec).sum()))
