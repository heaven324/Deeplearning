import numpy as np
from common.util import preprocessing


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocessing(text)

print(corpus)

print(id_to_word)


# 손으로 만들기

C = np.array([[0, 1, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 1, 1, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 0, 1, 0, 1, 0, 0],
              [0, 1, 0, 1, 0, 0, 0],
              [0, 1, 0, 0, 0, 0, 1],
              [0, 0, 0, 0, 0, 1, 0]], dtype = np.int32)

print(C[0]) # ID가 0인 벡터 표현

print(C[4]) # ID가 4인 벡터 표현

print(C[word_to_id['goodbye']]) # "goodbye"의 벡터표현

