import os
import cv2

path = "C:\\Users\\heaven\\Desktop\\imgdata\\000\\007"
    
file_list = os.listdir(path)
file_list1 = [path+'\\'+str(i) for i in file_list]
for j,i in enumerate(file_list1):
    img = cv2.imread(i)
    resize = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite("C:\\Users\\heaven\\python_project\\test_data\\total_image_32_part\\000\\" \
                + str(j+1+1546) + '.jpg',resize)