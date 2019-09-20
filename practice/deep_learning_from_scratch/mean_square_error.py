import numpy as np

def mean_square_error(y, t):
    return 0.5 * np.sum((y - t)**2)


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]    # 정답은 '2'

# 예1 : '2'일 확률이 가장 높다고 추정함 (0.6)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(mean_square_error(np.array(y), np.array(t)))

# 예1 : '7'일 확률이 가장 높다고 추정함 (0.6)
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(mean_square_error(np.array(y), np.array(t)))
