from util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)

c0 = C[word_to_id['you']] # [0 1 0 0 0 0 0]
c1 = C[word_to_id['i']]   # [0 1 0 1 0 0 0]

print(cos_similarity(c0, c1))
print(c0, c1)
