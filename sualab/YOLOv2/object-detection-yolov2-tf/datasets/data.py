import os
import numpy as np
from cv2 import imread, resize
import glob
import json

IM_EXTENSIONS = ['png', 'jpg', 'bmp']


def read_data(data_dir, image_size, pixels_per_grid=32, no_label=False):
    """
    Load the data and preprocessing for YOLO detector
    :param data_dir: str, path to the directory to read. It should include class_map, anchors, annotations
    :image_size: tuple, image size for resizing images
    :pixels_per_gird: int, the actual size of a grid
    :no_label: bool, whetehr to load labels
    :return: X_set: np.ndarray, shape: (N, H, W, C).
             y_set: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes).
    """
    im_dir = os.path.join(data_dir, 'images')
    class_map_path = os.path.join(data_dir, 'classes.json')
    anchors_path = os.path.join(data_dir, 'anchors.json')
    class_map = load_json(class_map_path)
    anchors = load_json(anchors_path)
    num_classes = len(class_map)
    grid_h, grid_w = [image_size[i] // pixels_per_grid for i in range(2)] # 예 : grid_h = 416/32, grid_w = 416/32
    im_paths = []
    for ext in IM_EXTENSIONS:   # 이미지들의 저장형식(png, jpg, bmp)순으로 모든 경로를 im_paths에 저장
        im_paths.extend(glob.glob(os.path.join(im_dir, '*.{}'.format(ext))))
    anno_dir = os.path.join(data_dir, 'annotations')
    images = []
    labels = []

    for im_path in im_paths:
        # load image and resize image
        im = imread(im_path)
        im = np.array(im, dtype=np.float32)
        im_origina_sizes = im.shape[:2]
        im = resize(im, (image_size[1], image_size[0]))    # 이미지 크기를 통일(예 : 416 x 416), squash해서 shape의 변형이 있을 수 있는 방법
        if len(im.shape) == 2:     # RGB채널이 아닌 흑백 이미지를 RGB채널 형식으로 통일
            im = np.expand_dims(im, 2)
            im = np.concatenate([im, im, im], -1)
        images.append(im)

        if no_label:
            labels.append(0)
            continue
        # load bboxes and reshape for yolo model
        name = os.path.splitext(os.path.basename(im_path))[0]     # 이미지 이름과 같은 .anno파일(annotation)을 찾기 위함
        anno_path = os.path.join(anno_dir, '{}.anno'.format(name))
        anno = load_json(anno_path)
        label = np.zeros((grid_h, grid_w, len(anchors), 5 + num_classes))   # label 기초 틀 
        for c_idx, c_name in class_map.items():
            if c_name not in anno:
                continue
            for x_min, y_min, x_max, y_max in anno[c_name]:
                oh, ow = im_origina_sizes
                # normalize object coordinates and clip the values
                x_min, y_min, x_max, y_max = x_min / ow, y_min / oh, x_max / ow, y_max / oh   # 좌표들을 0, 1 사이로 정규화
                x_min, y_min, x_max, y_max = np.clip([x_min, y_min, x_max, y_max], 0, 1)      # bbox가 이미지의 테두리를 벗어나지 않게 자름
                # assign the values to the best anchor
                anchor_boxes = np.array(anchors) / np.array([ow, oh])
                best_anchor = get_best_anchor(
                    anchor_boxes, [x_max - x_min, y_max - y_min])          # 5개의 anchor중 iou가 가장 높은 anchor의 인덱스를 가져옴
                cx = int(np.floor(0.5 * (x_min + x_max) * grid_w))
                cy = int(np.floor(0.5 * (y_min + y_max) * grid_h))
                label[cy, cx, best_anchor, 0:4] = [x_min, y_min, x_max, y_max]   # 검출할 객체의 센터가 위치한 그리드 자리에 bbox정보 저장
                label[cy, cx, best_anchor, 4] = 1.0                              # 검출할 객체의 센터가 위치한 그리드 자리에 검출할 책임을 저장  ?(confidence_score?)
                label[cy, cx, best_anchor, 5 + int(c_idx)] = 1.0                 # 검출할 객체의 센터가 위치한 그리드 자리에 class의 one_hot label 저장
        labels.append(label)
        # 최종적으로 labels는 그리드 형식의 이미지 모양에 bbox, best_anchor, one_hot class label 정보 저장

    X_set = np.array(images, dtype=np.float32)
    y_set = np.array(labels, dtype=np.float32)

    return X_set, y_set


def load_json(json_path):
    """
    Load json file
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_best_anchor(anchors, box_wh):
    """
    Select the best anchor with highest IOU
    """
    box_wh = np.array(box_wh)
    best_iou = 0
    best_anchor = 0
    for k, anchor in enumerate(anchors):   # iou를 계산해서 best_iou인 anchor를 선정
        intersect_wh = np.maximum(np.minimum(box_wh, anchor), 0.0)
        intersect_area = intersect_wh[0] * intersect_wh[1]
        box_area = box_wh[0] * box_wh[1]
        anchor_area = anchor[0] * anchor[1]
        iou = intersect_area / (box_area + anchor_area - intersect_area)
        if iou > best_iou:
            best_iou = iou
            best_anchor = k
    return best_anchor


class DataSet(object):

    def __init__(self, images, labels=None):
        """
        Construct a new DataSet object.
        :param images: np.ndarray, shape: (N, H, W, C)
        :param labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes).
        """
        if labels is not None:     # 라벨이 존재한다면 이미지의 개수와 라벨의 개수가 맞는지 확인
            assert images.shape[0] == labels.shape[0],\
                ('Number of examples mismatch, between images and labels')
        self._num_examples = images.shape[0]
        self._images = images
        self._labels = labels  # NOTE: this can be None, if not given.
        # image/label indices(can be permuted)
        self._indices = np.arange(self._num_examples, dtype=np.uint)     # 이미지 개수만큼의 인덱스 생성
        self._reset()

    def _reset(self):
        """Reset some variables."""
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    def sample_batch(self, batch_size, shuffle=True):
        """
        Return sample examples from this dataset.
        :param batch_size: int, size of a sample batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes)
        """

        if shuffle:
            indices = np.random.choice(self._num_examples, batch_size) # 배치사이즈만큼 무작위 이미지 선택
        else:
            indices = np.arange(batch_size)
        batch_images = self._images[indices]
        if self._labels is not None:                                   # 라벨이 있는 경우에 batch_images에 맞게 라벨을 선택
            batch_labels = self._labels[indices]
        else:
            batch_labels = None
        return batch_images, batch_labels

    def next_batch(self, batch_size, shuffle=True):
        """
        Return the next 'batch_size' examples from this dataset.
        :param batch_size: int, size of a single batch.
        :param shuffle: bool, whether to shuffle the whole set while sampling a batch.
        :return: batch_images: np.ndarray, shape: (N, H, W, C)
                 batch_labels: np.ndarray, shape: (N, g_H, g_W, anchors, 5 + num_classes)
        """

        start_index = self._index_in_epoch

        # Shuffle the dataset, for the first epoch
        if self._epochs_completed == 0 and start_index == 0 and shuffle: # 첫에폭에 shuffle=True 라면 모든 이미지 인덱스 shuffle
            np.random.shuffle(self._indices)

        # Go to the next epoch, if current index goes beyond the total number
        # of examples
        if start_index + batch_size > self._num_examples:
            # Increment the number of epochs completed
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start_index              # 현재 에폭에 남은 이미지 수
            indices_rest_part = self._indices[start_index:self._num_examples] # 그 이미지에 맞는 남은 인덱스 추출(rest_part)

            # Shuffle the dataset, after finishing a single epoch
            if shuffle:
                np.random.shuffle(self._indices)                              # 에폭이 끝난 관계로 다시 인덱스 shuffle

            # Start the next epoch
            start_index = 0                                                   # batch_size <= 전에폭 처리안된 이미지들 + 현에폭 처리할 이미지들
            self._index_in_epoch = batch_size - rest_num_examples             # 현에폭 처리할 이미지수 계산
            end_index = self._index_in_epoch
            indices_new_part = self._indices[start_index:end_index]           # 0부터 처리할 이미지 수 만큼(new_part)

            images_rest_part = self._images[indices_rest_part]                # 실제 이미지 지정(rest_part)
            images_new_part = self._images[indices_new_part]                  # 실제 이미지 지정(new_part)
            batch_images = np.concatenate(
                (images_rest_part, images_new_part), axis=0)                  # rest_part and new_part 이어붙이기
            if self._labels is not None:                                      # 이미지에 맞는 라벨 처리에 관한 if문
                labels_rest_part = self._labels[indices_rest_part]
                labels_new_part = self._labels[indices_new_part]
                batch_labels = np.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
            else:
                batch_labels = None
        else:                                              # 다음 에폭으로 넘어가는게 아니라면
            self._index_in_epoch += batch_size             # start_index를 저장해 두었으니 end_index로 변경
            end_index = self._index_in_epoch
            indices = self._indices[start_index:end_index] # batch_size만큼 현재 순서의 indices추출
            batch_images = self._images[indices]           # 그에 맞는 image지정
            if self._labels is not None:
                batch_labels = self._labels[indices]       # 그에 맞는 label지정(True일 경우)
            else:
                batch_labels = None

        return batch_images, batch_labels # 1. next epoch인 경우    code line 198, 203
                                          # 2. current epoch인 경우 code line 211, 213
