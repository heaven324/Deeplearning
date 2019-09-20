# batch 용

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    delta = 1e-7    # 마이너스 무한대가 발생하지 않게(log0은 마이너스 무한대)
    return -np.sum(t * np.log(y+delta)) / batch_size

