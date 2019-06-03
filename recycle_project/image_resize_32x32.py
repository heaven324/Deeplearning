import os
import cv2

path = "C:\\Users\\Administrator\\Desktop\\deeplearning_project\\imgdata\\total_image_224"
    
file_list = os.listdir(path)
file_name = sorted([int(i[:-4]) for i in file_list])
file_list1 = [path+'\\'+str(i)+'.jpg' for i in file_name]
for j,i in enumerate(file_list1):
    img = cv2.imread(i)
    resize = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("C:\\Users\\Administrator\\Desktop\\deeplearning_project\\imgdata\\total_image_32\\" \
                + str(j+1) + '.jpg',resize)