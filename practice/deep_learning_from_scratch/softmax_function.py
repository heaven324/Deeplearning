import numpy as np

def softmax(a):
    c = np.max(a)          # overflow를 막기위한 c
    exp_a = np.exp(x-c)    # overflow 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    
    return y

a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)            # softmax의 출력은 0에서 1사이
print(np.sum(y))    # sorfmax의 출력 총합은 1