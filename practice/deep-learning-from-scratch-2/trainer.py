import time
import numpy
import matplotlib.pyplot as plt
from common.np import *


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0
    
    def fit(self, x, t, max_epoch = 10, batch_size = 32, max_grad = None, eval_interval = 20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        
        start_time = time.time()
        for epoch in range(max_epoch):
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]
                
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)
                optimizer.update(params, grads)
                
                total_loss += loss
                loss_count += 1
                
                if iters % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print( '| epoch : %d | repeat : %d / %d | time : %d[s]| loss : %.2f'%\
                          (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1
    
    
    def plot(self, ylim = None):
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(numpy.arange(len(self.loss_list)), self.loss_list, label = 'train')
        plt.xlabel('repeat (x' + str(self.eval_interval)+')')
        plt.ylabel('loss')
        plt.show()
        
        
def remove_duplicate(params, grads):
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads
