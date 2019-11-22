# =============================================================================
# 1. cosine similarity = (x * y) / norm(x + ε) * norm(y + ε)
# 2. Positive Pointwise Mutual Information (양의 사용정보량)
#       PPMI = max(0, log_2{ (C(x,y)* N) / (C(x) * C(y)) })
#                   ( C(x, y) : 동시발생 횟수, C(x) : x의 등장 횟수 )
# =============================================================================

import os
from common.np import *


def preprocess(text):
    """
    preprocessing   2(page 85)
        - 여러 문장이 들어있는 텍스트 데이터들을 말뭉치로 전처리 하기 위한 함수

    1. text : 정제되지 않은 텍스트 데이터

    2. words : text 에 포함된 단어들의 리스트(순차적으로 저장됨)

    3. word_to_id : words의 단어(key)들에 id(value)를 부여한 딕셔너리 변수
                    (중복 없음)

    4. id_to_word : id(key)에 단어(value)들을 매칭한 변수

    5. corpus : words 자체를 활용하기 쉽지 않으므로 단어대신 id로 대체해서 
                생성한 말뭉치

    설명 : 데이터 내의 대문자를 소문자로 통일하고 '.' 과 같은 특수문자들을 분류
           하기 쉽게 ' .'로 대체한다.(정규표현식을 이용하면 더 쉽다.) 
           그 후 words라는 변수에 공백을 기준으로 단어들을 리스트화 한다. 
           그 후 단어들의 조작이 쉽게 단어에 id를 붙여 대응표를 
           딕셔너리(id_to_word, word_to_id)로 저장
    """
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    
    word_to_id = {}
    id_to_word = {}
    
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    
    corpus = np.array([word_to_id[w] for w in words])
    
    return corpus, word_to_id, id_to_word


def cos_similarity(x, y, eps = 1e-8):
    """
    Cosine Similarity   2(page 93)
        - 벡터(단어)간 유사도를 측정하는 함수

    1. x : 입력 벡터(단어)

    2. y : 입력 벡터(단어)

    3. eps : 0으로 나누어 지는 것을 방지하는 아주 작은 실수

    4. nx : x 나누기 x+eps 의 노름을 한 변수

    5. ny : y 나누기 y+eps 의 노름을 한 변수

    이론 : 1). cosine similarity = (x * y) / ∥x + ε∥ * ∥y + ε∥

    설명 : x와 y를 입력받아 두 변수의 nx, ny를 각각 구해서 곱한 수를 리턴
    """
    nx = x / np.sqrt(np.sum(x ** 2) + eps)
    ny = y / np.sqrt(np.sum(y ** 2) + eps)
    return np.dot(nx, ny)


def most_similar(query, word_to_id, id_to_word, word_matrix, top = 5):
    # 검색어
    if query not in word_to_id:
        print('%s 를 찾을 수 없습니다'%query)
        return 
    
    print('\n[query]' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # 코사인 유사도를 기준으로 내림차순으로 출력
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s'%(id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            return

        
def convert_one_hot(corpus, vocab_size):
    N = corpus.shape[0]

    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1

    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def create_co_matrix(corpus, vocab_size, window_size = 1):
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype = np.int32)
    
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - 1
            right_idx = idx + 1
            
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix


def ppmi(C, verbose = False, eps = 1e-8):
    M = np.zeros_like(C, dtype = np.float32)
    N = np.sum(C)
    S = np.sum(C, axis = 0)
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[i]*S[j]) + eps)
            M[i, j] = max(0, pmi)
            
            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% 완료'%(100*cnt/total))
    return M


def create_contexts_target(corpus, window_size = 1):
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
        
    return np.array(contexts), np.array(target)


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)

def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return np.asnumpy(x)


def clip_grads(grads, max_norm):
    """
    gradient clipping   2(page 246)
        - 기울기 폭발을 방지하기 위한 clipping함수

    1. grads : 모든 매개변수의 기울기들을 하나로 모은 변수(list나 array의 형태)

    2. max_norm : 기울기 총합의 문턱값(threshold)

    이론 : 1).  ∥ x ∥ = sqrt(x^2)
           2). rate = threshold / ∥ x ∥
           3). clipping = grads의 각 기울기에 rate를 곱한다.

    설명 : 기울기(grads)들의 총합의 노름(total_norm)을 구해 rate를 계산하고
           rate의 값이 1 보다 작다면 clipping을 한다.
    """
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate <1:
        for grad in grads:
            grad *= rate


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top = 5, answer = None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('%s(을)를 찾을 수 없습니다.'%word)
            return
    print('\n[analogy] ' + a + ':' + b + ' = ' + c + ':?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            return


def normalize(x):
    if x.ndim == 2:
        s = np.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = np.sqrt((x * x).sum())
        x /= s
    return x
