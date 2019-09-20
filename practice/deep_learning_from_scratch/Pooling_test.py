all_size_list = [784] + [100, 50] + [10]
print(all_size_list)

import sys, os
kk = sys.path
for i in kk:
    print(i)
sys.path.append(os.pardir)

input_dim = (1, 28, 28)
input_size = input_dim[1]
print(input_size)