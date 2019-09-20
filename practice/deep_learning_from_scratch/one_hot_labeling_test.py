# one hot labeling test
import numpy as np

x = np.array([0,1,2,3,4,5,6,7,8,9])
print('x\n', x)
t = np.zeros((x.size, 10))
print("t\n", t)
    
for idx,row in enumerate(t):
    print("idx, row \n", idx, row)
    row[x[idx]] = 1
    print(row)
print('----------------\n t = \n', t)