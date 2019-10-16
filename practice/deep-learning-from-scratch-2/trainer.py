import time
import numpy as np
import matplotlib.pyplot as plt

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
            idx = np.random.permutation(np.arange(data_size))
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters * batch_size : (iters + 1) * batch_size]
                batch_t = t[iters * batch_size : (iters + 1) * batch_size]
                
                loss = model.forward(batch_x, batch_t)
                model.backward()
                optimizer.update(model.params, model.grads)
                
                total_loss += loss
                loss_count += 1
                
                if (iters + 1) % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print( '| epoch : %d | repeat : %d / %d | time : %d[s]| loss : %.2f'%\
                          (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            self.current_epoch += 1
    
    
    def plot(self, ylim = None):
        plt.plot(np.arange(len(self.loss_list)), self.loss_list, label = 'train')
        plt.xlabel('repeat (x' + str(self.eval_interval)+')')
        plt.ylabel('loss')
        plt.show()