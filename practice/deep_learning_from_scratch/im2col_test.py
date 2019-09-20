# test 

a = np.array([[[[1, 2], [3, 4]], [[5,6], [7,8]]], [[[1, 2], [3, 4]], [[5,6], [7,8]]]])
print(a)

pad_img = np.pad(a, [(0,0), (0,0), (1, 1), (1, 1)], "constant")
print(pad_img)