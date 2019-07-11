from matplotlib import pyplot as plt
import cv2
import numpy as np
# video = cv2.VideoCapture("http://10.10.0.4:8080/video")
video = cv2.VideoCapture("http://10.10.0.4:4747/video")
ret = video.set(3,720)
ret = video.set(4,480)

while(True):
    ret, frame = video.read()    
#     print("type", type(frame))
    frame_expanded = np.expand_dims(frame, axis=0)
#     print(frame)
    cv2.imshow('Object detector', frame)
#     plt.imshow(frame)
    if cv2.waitKey(1) == ord('q'):
        break





    