# OrderedDict test
from collections import OrderedDict

layers = OrderedDict()
layers['a'] = [1,2,3]
layers['b'] = [4,5,6]
layers['c'] = [7,8,9]

print(layers, "\n")
print("layers.values()")
for i in layers.values():
    print(i)
print('')
print("layers.keys()")
for i in layers.keys():
    print(i)
