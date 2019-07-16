import numpy as np
import csv
import os
import cv2

def image_load(path):
    file_list = os.listdir(path)
    for i in range(len(file_list)):
        file_list[i] = int(file_list[i][0:-5])
    file_list.sort()
    for i in range(len(file_list)):
        file_list[i] = path + "\\" + str(file_list[i]) + ".jpeg"
    image =[]
    for i in file_list:
        img = cv2.imread(i)
        image.append(img)
    image = np.array(image)
    return image
    return file_list

def label_load(path):
    file = open(path)
    labeldata = csv.reader(file)
    labellist = []
    for i in labeldata:
        labellist.append(i)
    labellist = np.array(labellist).astype(int)
    labellist = np.eye(2)[labellist]
    return np.squeeze(labellist, axis = 1)

def next_batch(img, label, start, finish):
    return img[start:finish], label[start:finish]

def shuffle_batch(dataa, datab):
    x = np.arange(len(dataa))
    np.random.shuffle(x)
    data_list2 = dataa[x]
    label2 = datab[x]
    return data_list2, label2