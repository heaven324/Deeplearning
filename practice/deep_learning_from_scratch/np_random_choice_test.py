# np.random.choice test
import numpy as np

train_size = 1000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
print(batch_mask)

# batch_mask 는 batch 사이즈(10개)에 맞춰서 훈련 이미지의 인덱스(1~ 1000)
# 를 무작위로 추출 