# coding: utf-8

from common.np import *
import collections
from common.function_class import Embedding, SigmoidWithLoss


class EmbeddingDot:
    """
    Embedding Dot   2(page 166)
        - 다중 분류를 이진분류로 해결해기 위해 만들어진 계층으로
          정답레이블의 Embedding 계층과 dot(내적)의 처리를 합친 계층

    이론 : W_out은 단어들의 분산표현의 형태(행개수 = 어휘수)를 기반으로 
           만들어져있기 때문에 이진 분류로 바꾸기 위해 W_out에서 정답레이블의
           위치를 뽑아(target_W) h(은닉층 뉴런)와 dot을 계산(out )한다.
    """
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None
        
    def forward(self, h, idx):
        """
        W_out은 단어들의 분산표현의 형태(행개수 = 어휘수)를 기반으로 
        만들어져있기 때문에 이진 분류로 바꾸기 위해 W_out에서 정답레이블의
        위치를 뽑아(target_W) h(은닉층 뉴런)와 dot을 계산(out)한다.
        backward에서 사용하기 위해 변수 h와 target_W를 클래스변수 cache에 저장
        """
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis = 1)
        
        self.cache = (h, target_W)
        return out
    
    def backward(self, dout):
        """
        클래스변수 cache에서 h와 target_W를 받아서 곱셈의 역전파 이론으로 
        dtarget_W과 dh 를 구하고 dtarget_W를 embed에 backward시킨다.
            - 참고 그림 : fig 4-12.png
        """
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)
        
        dtarget_W = dout * h
        self.embed.backward(dtarget_W)
        dh = dout * target_W
        return dh


class UnigramSampler:   # 2(page 173)
    """
    nagative sampling을 할 때 확률분포에 따라 샘플링하게 해주는 클래스
    """
    def __init__(self, corpus, power, sample_size):
        """
        1. corpus : 단어 ID목록(단어의 구분은 index로)

        2. power : 확률 분포에 제곱할 값(낮은 확률의 단어를 구제하는 변수)

        3. sample_size(self) : 샘플링을 수행할 단어수 

        4. vocab_size : 어휘 수

        5. word_p(self) : 어휘별 확률분포(power적용)
        """
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None
        
        counts = collections.Counter()
        for word_id in corpus:
            counts[word_id] += 1
        
        vocab_size = len(counts)
        self.vocab_size = vocab_size
        
        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]
            
        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)
        
    
    def get_negative_sample(self, target):
        """
        1. target : 긍정적인 예로 해석한 단어

        init에서 계산한 확률분포(word_p)에 따라 np.random.choice를 이용해 sample_size
        만큼의 부정적인 예를 리턴해주는 함수(배치처리 가능)
        """
        batch_size = target.shape[0]

        if not GPU:
            negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

            for i in range(batch_size):
                p = self.word_p.copy()
                target_idx = target[i]
                p[target_idx] = 0
                p /= p.sum()
                negative_sample[i, :] = np.random.choice(self.vocab_size, size=self.sample_size, replace=False, p=p)
        else:
            # GPU(cupy）로 계산할 때는 속도를 우선한다.
            # 부정적 예에 타깃이 포함될 수 있다.
            negative_sample = np.random.choice(self.vocab_size, size=(batch_size, self.sample_size),
                                               replace=True, p=self.word_p)

        return negative_sample


class NegativeSamplingLoss:   # 2(page 174)
    def __init__(self, W, corpus, power = 0.75, sample_size = 5):
        """
        1. W : 출력층의 가중치(W_out)

        2. corpus : 단어 ID의 리스트

        3. power : 부정 단어 추출에서 확률 분포에 제곱할 값
                   (낮은 확률의 단어를 구제하는 변수)

        4. sample_size : 부정 샘플링할 단어 수(긍정 + 부정 단어만큼 layer 생성)

        5. sampler : UnigramSampler 클래스를 담은 변수

        6. loss_layers : SigmoidWithLoss 클래스를 sample_size + 1 만큼 담은 리스트 변수

        7. embed_dot_layers : EmbeddingDot 클래스를 sample_size + 1 만큼 담은 리스트 변수

        8. params, grads : embed_dot_layers 리스트에 담긴 EmbeddingDot 클래스 들의
                           params와 grads를 한데 모아 담은 변수
                           (단순 객체복사를 위해 변경가능 객체(list)들이 담겨있다)
        """
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.loss_layers = [SigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]
        
        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads
            
    
    def forward(self, h, target):
        """
        init에서 정의한 클래스들을 이용해 부정단어를 샘플링하고 
        순서대로 forward를 진행하는데 score_label이 긍정(1)과 부정(0)이 다르기 때문에
        따로 진행 후 loss 를 출력
        """
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)
        
        # positive forward
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype = np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)
        
        negative_label = np.zeros(batch_size, dtype = np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i].forward(h,negative_target)
            loss += self.loss_layers[1 + i].forward(score, negative_label)
        
        return loss
    
    
    def backward(self, dout = 1):
        """
        init에서 정의한 클래스들 각각 순서대로 backward를 진행
        """
        dh = 0
        for l0, l1 in zip(self.loss_layers, self.embed_dot_layers):
            dscore = l0.backward(dout)
            dh += l1.backward(dscore)
            
        return dh