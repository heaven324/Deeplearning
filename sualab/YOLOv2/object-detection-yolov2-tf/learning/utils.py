import os
import os.path as osp
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from learning.colors import COLORS
import cv2
import json
from imageio import imsave


def maybe_mkdir(*directories):
    for d in directories:
        if not osp.isdir(d):
            os.mkdir(d)


def plot_learning_curve(exp_idx, step_losses, step_scores, eval_scores=None,
                        mode='max', img_dir='.'):
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    axes[0].plot(np.arange(1, len(step_losses) + 1), step_losses, marker='')
    axes[0].set_ylabel('loss')
    axes[0].set_xlabel('Number of iterations')
    axes[1].plot(np.arange(1, len(step_scores) + 1),
                 step_scores, color='b', marker='')
    if eval_scores is not None:
        axes[1].plot(np.arange(1, len(eval_scores) + 1),
                     eval_scores, color='r', marker='')
    if mode == 'max':
        axes[1].set_ylim(0.5, 1.0)
    else:    # mode == 'min'
        axes[1].set_ylim(0.0, 0.5)
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Number of epochs')

    # Save plot as image file
    plot_img_filename = 'learning_curve-result{}.svg'.format(exp_idx)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, plot_img_filename))

    # Save details as pkl file
    pkl_filename = 'learning_curve-result{}.pkl'.format(exp_idx)
    with open(os.path.join(img_dir, pkl_filename), 'wb') as fo:
        pkl.dump([step_losses, step_scores, eval_scores], fo)


def nms(boxes, conf_thres=0.2, iou_thres=0.5):
    '''
    params : x1, y1, x2, y2, scores shape: (box개수, )
    '''
    x1 = boxes[..., 0]             # 각 좌표값들
    y1 = boxes[..., 1]
    x2 = boxes[..., 2]
    y2 = boxes[..., 3]
    areas = (x2 - x1) * (y2 - y1)  # 각 bbox 넓이 계산
    scores = boxes[..., 4]         # conf score(?)

    keep = []
    order = scores.argsort()[::-1] # scores에서 가장 큰 값부터 인덱스를 반환

    while order.size > 0:
        i = order[0]
        keep.append(i)                         # 현재 스코어가 가장 높은 인덱스 keep
        '''
        밑의 코드 해석
        xy좌표계에서 1사분면에 box 두개가 있다고 가정해보면 각 박스에서 x1 < x2, y1 < y2인 상태
        여기서 intersection 부분을 찾아야 하기 때문에 두 박스의 x1좌표중 큰 값을 xx1 y1도 마찬가지
        x2좌표중 작은 값을 xx2 y2도 마찬가지로 해서 박스를 그리면 intersection이 나옴

        그렇게 구한 intersection이 겹치는 부분이 없을 경우를 대비해서 w, h에 0.0의 최소값을 적용
        '''
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter) # IOU계산    * 박스 개수가 원래보다 하나 줄어든다(가장 큰 박스를 비교대상자로 써서)

        inds = np.where(ovr <= iou_thres)[0]                # 비교한 박스들 중 많이 안겹친 박스(0.5 이하)들의 인덱스 추출, [0]의 의미는 껍데기 벗기기
                                                            # 여기서 inds들은 ovr계산할때 한칸씩 앞으로 땡겨졌음
        order = order[inds + 1]                             # + 1은 한칸씩 땡겨진 인덱스를 order에 맞추기 위함
                                                            # 위 식의 결과로 많이 안겹친 박스들 중 스코어 내림차순으로 order가 만들어져서 다음 루프에 들어감

    nms_box = []
    for idx in range(len(boxes)):                      # 모든 box들의 인덱스 순환
        if idx in keep and boxes[idx, 4] > conf_thres: # while문으로 keep한 것들을 conf문턱값으로 한번 더 걸러 남은 것들 box정보 저장
            nms_box.append(boxes[idx])
        else:
            nms_box.append(np.zeros(boxes.shape[-1]))  # 걸러진 box들은 0으로채워진 (5 + num_classes, )모양으로 추가
    boxes = np.array(nms_box)
    return boxes   # boxs: ndarray, shape: (box개수, 5 + num_classes) shape의 변화가 없음


def convert_boxes(input_y):
    '''
     input_y shape: 배치일 경우 (N, g_H, g_W, anchors, 5 + num_classes), 아닐경우 4차원
    '''
    is_batch = len(input_y.shape) == 5                                     # is_batch : Bool
    if not is_batch:                                                       # input_y가 5차원이 아니라면(4차원이라면)
        input_y = np.expand_dims(input_y, 0)                               # 차원을 하나 늘린다(axis = 0) : 5차원
    boxes = np.reshape(input_y, (input_y.shape[0], -1, input_y.shape[-1])) # boxs : (N, box수, 5 + num_classes)
    if is_batch:
        return np.array(boxes)
    else:
        return boxes[0]                                                    # 껍데기 벗기기


def predict_nms_boxes(input_y, conf_thres=0.2, iou_thres=0.5):
    '''
     input_y shape: 배치일 경우 (N, g_H, g_W, anchors, 5 + num_classes), 아닐경우 4차원
    '''
    is_batch = len(input_y.shape) == 5                                     # is_batch : Bool
    if not is_batch:                                                       # input_y가 5차원이 아니라면(4차원이라면)
         input_y = np.expand_dims(input_y, 0)                              # 차원을 하나 늘린다(axis = 0) : 5차원
    boxes = np.reshape(input_y, (input_y.shape[0], -1, input_y.shape[-1])) # boxs : (N, box수, 5 + num_classes)
    nms_boxes = []
    for box in boxes:                                                      # box : (box개수, 5 + num_classes)
        nms_box = nms(box, conf_thres, iou_thres)                          # nms_box: ndarray, shape: (box개수, 5 + num_classes)
        nms_boxes.append(nms_box)                                          # nms_boxs: list, shape: (batch수, box개수, 5 + num_classes)
    if is_batch:
        return np.array(nms_boxes)
    else:
        return nms_boxes[0]                                                # 껍데기 벗기기


def cal_recall(gt_bboxes, bboxes, iou_thres=0.5):
    '''
    gt_bboxs shape: (N, 0포함 true_box개수, 5 + num_classes) : labeling된 박스들
    bboxes shape: (N, 0포함 pred_box개수, 5 + num_classes) : pred되어 만들어진 박스들
    '''
    p = 0   # 전체 오브젝트 수
    tp = 0  # 위치를 찾고 올바르게 분류한 오브젝트 수
    for idx, (gt, bbox) in enumerate(zip(gt_bboxes, bboxes)): # gt, bbox: shape: (각 box개수, 5 + num_classes), 한 이미지씩을 의미하는 for문
        gt = gt[np.nonzero(np.any(gt > 0, axis=1))]           # 실제 데이터가 있는 박스들만 추출
        bbox = bbox[np.nonzero(np.any(bbox > 0, axis=1))]     # 같은 방식
        p += len(gt)                                          # 한 이미지에서의 true박스(오브젝트) 개수 추가
        if bbox.size == 0:
            continue
        iou = _cal_overlap(gt, bbox)                          # iou shape : (true_box개수, pred_box개수)
        predicted_class = np.argmax(bbox[...,5:], axis=-1)    # 각 pred_box별 예측된 클래스 배열 shape: (pred_box개수, )
        for g, area in zip(gt, iou):                                      # 한 true박스씩을 의미하는 for문
            gt_c = np.argmax(g[5:])                                       # 현 true박스 정답 클래스의 index
            idx = np.argmax(area)                                         # 현 true박스와 가장 높은 iou를 가진 pred박스의 index
            if np.max(area) > iou_thres and predicted_class[idx] == gt_c: # iou문턱값보다 크고 예측된 클래스가 정답이라면
                tp += 1                                                   # 위치를 찾고 올바르게 분류한 오브젝트 1개 추가
    return tp / p

def cal_F1(gt_bboxes, bboxes, iou_thres=0.5):
    p = 0
    tp = 0
    pp = 0
    for idx, (gt, bbox) in enumerate(zip(gt_bboxes, bboxes)):
        gt = gt[np.nonzero(np.any(gt > 0, axis=1))]
        bbox = bbox[np.nonzero(np.any(bbox > 0, axis=1))]
        p += len(gt)
        pp += len(bbox)
        if bbox.size == 0:
            continue
        iou = _cal_overlap(gt, bbox)
        predicted_class = np.argmax(bbox[...,5:], axis=-1)
        for g, area in zip(gt, iou):
            gt_c = np.argmax(g[5:])
            idx = np.argmax(area)
            if np.max(area) > iou_thres and predicted_class[idx] == gt_c:
                tp += 1
    recall = tp / p
    precision = tp / pp

    f1 = 2*precision*recall / (precision + recall)

    return f1

def _cal_overlap(a, b):
    '''
    a : gt shape : (0이상 box개수, 5 + num_classes)   : labeling
    b : bbox shape : (0이상 box개수, 5 + num_classes) : predict
    '''
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])              # pred된 박스들의 면적 계산
                                                                  # 아래 계산에서 expand_dims를 하는 이유는 broadcasting을 사용하기 위함
    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - \
        np.maximum(np.expand_dims(a[:, 0], axis=1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - \
        np.maximum(np.expand_dims(a[:, 1], axis=1), b[:, 1])

    iw = np.maximum(iw, 0)                                        # 길이에 -값은 올 수 없기 때문
    ih = np.maximum(ih, 0)
    intersection = iw * ih

    ua = np.expand_dims((a[:, 2] - a[:, 0]) *
                        (a[:, 3] - a[:, 1]), axis=1) + area - intersection

    ua = np.maximum(ua, np.finfo(float).eps)

    return intersection / ua # shape: (a.shape[0], b.shape[0])

def draw_pred_boxes(image, pred_boxes, class_map, text=True, score=False):
    im_h, im_w = image.shape[:2]
    output = image.copy()
    for box in pred_boxes:
        overlay = output.copy()

        class_idx = np.argmax(box[5:])
        color = COLORS[class_idx]
        line_width, alpha = (2, 0.8)
        x_min, x_max = [int(x * im_w) for x in [box[0], box[2]]]
        y_min, y_max = [int(x * im_h) for x in [box[1], box[3]]]
        cv2.rectangle(overlay, (x_min, y_min),
                      (x_max, y_max), color, line_width)
        output = cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0)

        if text:
            p_text = str(round(np.max(box[5:]), 3)) if score else class_map[str(class_idx)]
            y_offset = -6
            text_size = 0.6
            text_line_width = 1
            output = cv2.putText(output, p_text, (x_min + 4, y_min + y_offset),
                                 cv2.FONT_HERSHEY_DUPLEX, text_size, color, text_line_width)

    return output


def iou(b1, b2):
    l1, l2 = b1[0] - int(0.5 * b1[2]), b2[0] - int(0.5 * b2[2])
    u1, u2 = l1 + b1[2] - 1, l2 + b2[2] - 1
    intersection_w = max(0, min(u1, u2) - max(l1, l2) + 1)
    if intersection_w == 0:
        return 0
    l1, l2 = b1[1] - int(0.5 * b1[3]), b2[1] - int(0.5 * b2[3])
    u1, u2 = l1 + b1[3] - 1, l2 + b2[3] - 1
    intersection_h = max(0, min(u1, u2) - max(l1, l2) + 1)
    intersection = intersection_w * intersection_h
    if intersection == 0:
        return 0

    union = b1[2] * b1[3] + b2[2] * b2[3] - intersection
    if union == 0:
        raise ValueError('Union value must not be a zero or negative number. (boxes: {}, {})'.format(b1, b2))

    return intersection / union


def kmeans_iou(boxes, k, n_iter=10):
    n_boxes = len(boxes)
    if k > n_boxes:
        raise ValueError('k({}) must be less than or equal to the number of boxes({}).'.format(k, n_boxes))

    # Update clusters and centroids alternately.
    centroids = boxes[np.random.choice(n_boxes, k, replace=False)]
    for _ in range(n_iter):
        cluster_indices = np.array([np.argmax([iou(b, c) for c in centroids]) for b in boxes])
        for i in range(k):
            if np.count_nonzero(cluster_indices == i) == 0:
                print(i)
        centroids = [np.mean(boxes[cluster_indices == i], axis=0) for i in range(k)]

    return np.array(centroids)


def calculate_anchor_boxes(im_paths, anno_dir, num_anchors):
    boxes = []

    for im_path in im_paths:
        im_h, im_w = scipy.misc.imread(im_path).shape[:2]

        anno_fname = '{}.anno'.format(osp.splitext(osp.basename(im_path))[0])
        anno_fpath = osp.join(anno_dir, anno_fname)
        if not osp.isfile(anno_fpath):
            print('ERROR | Corresponding annotation file is not found: {}'.format(anno_fpath))
            return

        with open(anno_fpath, 'r') as f:
            anno = json.load(f)
        for class_name in anno:
            for x_min, y_min, x_max, y_max in anno[class_name]:
                center_x, center_y = (x_max + x_min) * 0.5, (y_max + y_min) * 0.5
                width, height = x_max - x_min + 1, y_max - y_min + 1
                boxes.append([center_x, center_y, width, height])

    boxes = np.array(boxes, dtype=np.float32)
    anchors = kmeans_iou(boxes, num_anchors, 10)[:, 2:]

    return np.array(anchors)
