#!/usr/bin/env python
# coding: utf-8

# Module 사용을 위한 주피터 세션 세팅
# - TF SLIM & OBJECT_DETECTOION API

# In[11]:


# TF SLIM과 OBJECT_DETECTOION API를 사용하기 위한 PYTHON_PATH APPEND
# 최초 1회만 수행
import sys
import os

# !export PYTHONPATH=$PYTHONPATH:/home/xerato/workspace/tmp/models:/home/xerato/workspace/tmp/models/research:/home/xerato/workspace/tmp/models/research/slim
PATH_OBJ_DETECTION="/home/xerato/workspace/models/research/object_detection/"            

sys.path.append("/home/xerato/workspace/models/research/object_detection/")
sys.path.append("/home/xerato/workspace/models/research/")


# 파이썬과 골치아픈 두녀석의 버전을 체크합니다.

# In[12]:


import cv2
print("cv2", cv2.__version__)
import tensorflow as tf
print("tensorflow", tf.__version__)


# In[13]:


import os
import numpy as np
import sys
import datetime
from matplotlib import pyplot as plt
from utils import label_map_util
from utils import visualization_utils as vis_util


# In[14]:


MODEL_NAME = 'faster_rcnn_inception_v2' # 후일 모델명에 맞게 스위치 할 수 있게 구현합니다.
PATH_TO_FROZEN_GRAPH = os.path.join(PATH_OBJ_DETECTION, 'inference_graph','frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(PATH_OBJ_DETECTION, 'training', 'labelmap.pbtxt')
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# In[15]:


NUM_CLASSES = 25

# item {  name: "Plastic_0"  id: 1 }
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)

# [{'id': 1, 'name': 'Plastic_0'},
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

# {1: {'id': 1, 'name': 'Plastic_0'},
category_index = label_map_util.create_category_index(categories)


# In[16]:


# 학습된 그래프를 설정

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
    sess = tf.Session(graph=detection_graph)
    print("sess1", sess)
    
print("sess2", sess)


# In[17]:


print("URL 예1 : http://0.0.0.0:8080/video")
print("URL 예1 : http://0.0.0.0:4747/mjpegfeed?1280x720")
input_url = input("stream URL : ")
# input_url = "http://10.10.0.4:8080/video"


# 그래프로부터 세션함수 설정

# In[18]:



def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
  
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        # 우리는 4개의 속성을 사용합니다.
        # 'num_detections', 'detection_boxes', 'detection_scores','detection_classes'
        
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])

        
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)

        
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})
    
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


# In[19]:




image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')


# 웹캠(혹은 비디오) 설정

# In[20]:


# video = cv2.VideoCapture(input_url)


# In[21]:


# %matplotlib inline

def start_streaming():
    print("start_streaming")
    video = cv2.VideoCapture(input_url)
    ret = video.set(3,720)
    ret = video.set(4,480)
    try:
        while(True):
            ret, frame = video.read()        
            frame_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.70)

            cv2.imshow('NAMU', frame)
            # 키지정 종료 
            if cv2.waitKey(1) == ord('q'):
               break
    except Exception as trace:
        print(trace)
    finally:
        print("종료")
        video.release()


# In[22]:


# %matplotlib inline

def start_streaming_snapshot(seconds=5):
    # N초 마다 스냅샷
    video = cv2.VideoCapture(input_url)
    ret = video.set(3,720)
    ret = video.set(4,480)
    try:
        while(True):
            ret, frame = video.read()        
            frame_expanded = np.expand_dims(frame, axis=0)

            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: frame_expanded})

            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)

            if datetime.datetime.now().second % seconds == 0 :
                print(datetime.datetime.now().second)

                plt.figure(figsize=(20,20))
                plt.imshow(frame)
                plt.show()


    except Exception as trace:
        print(trace)
    finally:
        print("종료")
        video.release()


# In[25]:


# start_streaming_snapshot()


# In[23]:


# start_streaming_snapshot()


# In[26]:


# video.release()


# In[ ]:





# In[4]:


def main():
    start_streaming()


if __name__ == '__main__':
    print("main 진입")
    main()


# adb 연결 디바이스 확인
# 
# xerato@woody:~/workspace/deep_namu_2$ adb devices
# 
# List of devices attached
# 
# ce10171a78615030017e	device
# 
# 
# adb 포트포워딩
# 
# xerato@woody:~/workspace/deep_namu_2$ adb forward tcp:8080 tcp:8080

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[2]:





# In[ ]:


# start_streaming()


# In[5]:


#get_ipython().system('jupyter nbconvert --to script namu_via_streaming.ipynb')


# In[ ]:




