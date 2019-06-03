import loader3

train_image = "C:\\Users\\heaven\\python_project\\test_data\\total_image_32_part\\000"
train_label = "C:\\Users\\heaven\\python_project\\test_data\\total_image_32_part\\label000.csv"

trainX = loader3.image_load(train_image)
trainY = loader3.label_load(train_label)


def next_batch(img, label, start, finish):
    return img[start:finish], label[start:finish]

trainx, trainy = next_batch(trainX, trainY, 0, 100)

print(trainX.shape)
print(trainY.shape)
print(trainx.shape)
print(trainy.shape)