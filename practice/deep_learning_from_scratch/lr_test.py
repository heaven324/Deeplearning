def test(lr = 0.0001):
    return lr

a = test(lr = 0.1)
print(a)
optimizer_param={'lr':0.01}
b = test(**optimizer_param)
# print(**optimizer_param)

print(b)