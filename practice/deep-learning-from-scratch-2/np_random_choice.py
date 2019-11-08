import numpy as np

print(np.random.choice(10))  # 5
print(np.random.choice(10))  # 9


# words에서 하나만 무작위로 샘플링
words = ['you', 'say', 'goodbye', 'I', 'hello', '.']
print(np.random.choice(words))


# 5개만 무작위로 샘플링(중복 있음)
print(np.random.choice(words, size = 5))


# 5개만 무작위로 샘플링(중복 없음)
print(np.random.choice(words, size = 5, replace = False))


# 확률분포에 따라 샘플링
p = [0.5, 0.1, 0.05, 0.2, 0.05, 0.1]
print(np.random.choice(words, p = p))



# 0.75제곱을 하는 이유( 빈도가 낮은 단어의 확률을 살짝 높이기 위해서)
p = [0.7, 0.29, 0.01]
new_p = np.power(p, 0.75)
print(new_p)
new_p /= np.sum(new_p)
print(new_p)

