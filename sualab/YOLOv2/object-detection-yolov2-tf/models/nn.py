import time
from abc import abstractmethod, ABCMeta
import tensorflow as tf
import numpy as np
from models.layers import conv_layer, max_pool, fc_layer, batchNormalization


class DetectNet(metaclass=ABCMeta):
    """Base class for Convolutional Neural Networks for detection."""

    def __init__(self, input_shape, num_classes, **kwargs):
        """
        model initializer
        :param input_shape: tuple, shape (H, W, C)
        :param num_classes: int, total number of classes
        """
        self.X = tf.placeholder(tf.float32, [None] + input_shape) # 배치를 포함한 input 공간 생성, X shape : (None, 416, 416, 3)
        self.is_train = tf.placeholder(tf.bool)
        self.num_classes = num_classes                            # 모델이 감지해야할 class 개수
        self.d = self._build_model(**kwargs)                      # class 호출시 모델을 인스턴스 변수로 저장
        self.pred = self.d['pred']
        self.loss = self._build_loss(**kwargs)                    # class 호출시 loss를 인스턴스 변수로 저장

    @abstractmethod
    def _build_model(self, **kwargs):
        """
        Build model.
        This should be implemented.
        """
        pass

    @abstractmethod
    def _build_loss(self, **kwargs):
        """
        build loss function for the model training.
        This should be implemented.
        """
        pass

    def predict(self, sess, dataset, verbose=False, **kwargs):
        """
        Make predictions for the given dataset.
        :param sess: tf.Session.
        :param dataset: DataSet.
        :param verbose: bool, whether to print details during prediction.
        :param kwargs: dict, extra arguments for prediction.
                -batch_size: int, batch size for each iteration.
        :return _y_pred: np.ndarray, shape: shape of self.pred
        """

        batch_size = kwargs.pop('batch_size', 16) # 배치 크기(기본 16)

        num_classes = self.num_classes            # 모델이 감지해야할 class 개수
        pred_size = dataset.num_examples          # 주어진 dataset의 이미지 개수
        num_steps = pred_size // batch_size       # dataset을 한바퀴 도는데 걸리는 for문의 step
        flag = int(bool(pred_size % batch_size))  # 나누어 떨어진다면 False, else True
        if verbose:
            print('Running prediction loop...')

        # Start prediction loop
        _y_pred = []                                              # 전체 이미지의 pred결과 모음
        start_time = time.time()
        for i in range(num_steps + flag):
            if i == num_steps and flag:                           # batch_size와 크기가 다른 마지막 루프가 오면
                _batch_size = pred_size - num_steps * batch_size  # batch_size크기 조정
            else:
                _batch_size = batch_size                          # 그게 아니라 보통 루프이면 그대로
            X, _ = dataset.next_batch(_batch_size, shuffle=False) # pred에 사용 될 이미지 할당

            # Compute predictions
            # (N, grid_h, grid_w, 5 + num_classes)
            y_pred = sess.run(self.pred_y, feed_dict={
                              self.X: X, self.is_train: False})   # predict 수행 (from nn.py line 395)

            _y_pred.append(y_pred)                                # pred된 결과 모음

        if verbose:                                               # 출력 여부
            print('Total prediction time(sec): {}'.format(
                time.time() - start_time))

        _y_pred = np.concatenate(_y_pred, axis=0)                 # 배치 단위로 잘려있는(axis = 0) 결과값 연결하기
        return _y_pred


class YOLO(DetectNet):
    """YOLO class"""

    def __init__(self, input_shape, num_classes, anchors, **kwargs):

        self.grid_size = grid_size = [x // 32 for x in input_shape[:2]] # grid_size : [13, 13]
        self.num_anchors = len(anchors)                                 # num_anchors : 5
        self.anchors = anchors                                          # anchors shape : (5, 2)
        self.y = tf.placeholder(tf.float32, [None] +
                                [self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + num_classes]) # y : float32, shape : (None, 13, 13, 5, 6)
        super(YOLO, self).__init__(input_shape, num_classes, **kwargs)  # 부모(DetectNet)클래스 변수 초기화

    def _build_model(self, **kwargs):
        """
        Build model.
        :param kwargs: dict, extra arguments for building YOLO.
                -image_mean: np.ndarray, mean image for each input channel, shape: (C,).
        :return d: dict, containing outputs on each layer.

        * Darknet(layer1 ~ layer19)
        layer1 [conv1  - batch_norm1  - leaky_relu1  - pool1 ] ->
        layer2 [conv2  - batch_norm2  - leaky_relu2  - pool2 ] ->
        layer3 [conv3  - batch_norm3  - leaky_relu3          ] ->
        layer4 [conv4  - batch_norm4  - leaky_relu4          ] ->
        layer5 [conv5  - batch_norm5  - leaky_relu5  - pool5 ] ->
        layer6 [conv6  - batch_norm6  - leaky_relu6          ] ->
        layer7 [conv7  - batch_norm7  - leaky_relu7          ] ->
        layer8 [conv8  - batch_norm8  - leaky_relu8  - pool8 ] ->
        layer9 [conv9  - batch_norm9  - leaky_relu9          ] ->
        layer10[conv10 - batch_norm10 - leaky_relu10         ] ->
        layer11[conv11 - batch_norm11 - leaky_relu11         ] ->
        layer12[conv12 - batch_norm12 - leaky_relu12         ] ->
        layer13[conv13 - batch_norm13 - leaky_relu13 - pool13] ->
        layer14[conv14 - batch_norm14 - leaky_relu14         ] ->
        layer15[conv15 - batch_norm15 - leaky_relu15         ] ->
        layer16[conv16 - batch_norm16 - leaky_relu16         ] ->
        layer17[conv17 - batch_norm17 - leaky_relu17         ] ->
        layer18[conv18 - batch_norm18 - leaky_relu18         ] ->
        layer19[conv19 - batch_norm19 - leaky_relu19         ] ->

        * YOLO Detector(layer20 ~ layer22)
        layer20[conv20 - batch_norm20 - leaky_relu20         ] ->

        layer21[skip_connection(leaky_relu13) - skip_batch - skip_leaky_relu - 
                skip_space_to_depth_x2 - concat21(skip_space_to_depth_x2, leaky_relu20)] ->

        layer22[conv22 - batch_norm22 - leaky_relu22         ] -> logit(conv) -> pred(reshape)
        """

        d = dict()                             # return dict
        x_mean = kwargs.pop('image_mean', 0.0) # train.py에서 제공된 **kwargs변수가 없음 -> x_mean = 0.0

        # input
        X_input = self.X - x_mean # X_input shape : (None, 416, 416, 3), x_mean을 뺀다는게 무슨 의미지?
                                  # x_mean을 뺀다는 것은 모든 이미지 숫자의 평균을 각 픽셀마다 빼서 평균이 0인 이미지들로 만든다는 것 같음
        is_train = self.is_train  # 부모클래스에선 False(validation set 통과하는 것이기 때문), optimizers.py 에선 True(training을 진행하기 때문)

        #conv1 - batch_norm1 - leaky_relu1 - pool1
        with tf.variable_scope('layer1'):
            d['conv1'] = conv_layer(X_input, 3, 1, 32,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 416, 416, 3),  out shape :(B, 416, 416, 32)
            d['batch_norm1'] = batchNormalization(d['conv1'], is_train)                  # input modify for activation layer
            d['leaky_relu1'] = tf.nn.leaky_relu(d['batch_norm1'], alpha=0.1)
            d['pool1'] = max_pool(d['leaky_relu1'], 2, 2, padding='SAME')                # in shape :(B, 416, 416, 32), out shape :(B, 208, 208, 32)
        # (416, 416, 3) --> (208, 208, 32)
        print('layer1.shape', d['pool1'].get_shape().as_list())

        #conv2 - batch_norm2 - leaky_relu2 - pool2
        with tf.variable_scope('layer2'):
            d['conv2'] = conv_layer(d['pool1'], 3, 1, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 208, 208, 32), out shape :(B, 208, 208, 64)
            d['batch_norm2'] = batchNormalization(d['conv2'], is_train)                  # input modify for activation layer
            d['leaky_relu2'] = tf.nn.leaky_relu(d['batch_norm2'], alpha=0.1)
            d['pool2'] = max_pool(d['leaky_relu2'], 2, 2, padding='SAME')                # in shape :(B, 208, 208, 64), out shape :(B, 104, 104, 64)
        # (208, 208, 32) --> (104, 104, 64)
        print('layer2.shape', d['pool2'].get_shape().as_list())

        #conv3 - batch_norm3 - leaky_relu3
        with tf.variable_scope('layer3'):
            d['conv3'] = conv_layer(d['pool2'], 3, 1, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 104, 104, 64), out shape :(B, 104, 104, 128)
            d['batch_norm3'] = batchNormalization(d['conv3'], is_train)                  # input modify for activation layer
            d['leaky_relu3'] = tf.nn.leaky_relu(d['batch_norm3'], alpha=0.1)
        # (104, 104, 64) --> (104, 104, 128)
        print('layer3.shape', d['leaky_relu3'].get_shape().as_list())

        #conv4 - batch_norm4 - leaky_relu4
        with tf.variable_scope('layer4'):
            d['conv4'] = conv_layer(d['leaky_relu3'], 1, 1, 64,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 104, 104, 128), out shape :(B, 104, 104, 64)
            d['batch_norm4'] = batchNormalization(d['conv4'], is_train)                  # input modify for activation layer
            d['leaky_relu4'] = tf.nn.leaky_relu(d['batch_norm4'], alpha=0.1)
        # (104, 104, 128) --> (104, 104, 64)
        print('layer4.shape', d['leaky_relu4'].get_shape().as_list())

        #conv5 - batch_norm5 - leaky_relu5 - pool5
        with tf.variable_scope('layer5'):
            d['conv5'] = conv_layer(d['leaky_relu4'], 3, 1, 128,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 104, 104, 64),  out shape :(B, 104, 104, 128)
            d['batch_norm5'] = batchNormalization(d['conv5'], is_train)                  # input modify for activation layer
            d['leaky_relu5'] = tf.nn.leaky_relu(d['batch_norm5'], alpha=0.1)
            d['pool5'] = max_pool(d['leaky_relu5'], 2, 2, padding='SAME')                # in shape :(B, 104, 104, 128), out shape :(B, 52, 52, 128)
        # (104, 104, 64) --> (52, 52, 128)
        print('layer5.shape', d['pool5'].get_shape().as_list())

        #conv6 - batch_norm6 - leaky_relu6
        with tf.variable_scope('layer6'):
            d['conv6'] = conv_layer(d['pool5'], 3, 1, 256,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 52, 52, 128), out shape :(B, 52, 52, 256)
            d['batch_norm6'] = batchNormalization(d['conv6'], is_train)                  # input modify for activation layer
            d['leaky_relu6'] = tf.nn.leaky_relu(d['batch_norm6'], alpha=0.1)
        # (52, 52, 128) --> (52, 52, 256)
        print('layer6.shape', d['leaky_relu6'].get_shape().as_list())

        #conv7 - batch_norm7 - leaky_relu7
        with tf.variable_scope('layer7'):
            d['conv7'] = conv_layer(d['leaky_relu6'], 1, 1, 128,
                                    padding='SAME', weights_stddev=0.01, biases_value=0.0) # in shape :(B, 52, 52, 256), out shape :(B, 52, 52, 128)
            d['batch_norm7'] = batchNormalization(d['conv7'], is_train)                    # input modify for activation layer
            d['leaky_relu7'] = tf.nn.leaky_relu(d['batch_norm7'], alpha=0.1)
        # (52, 52, 256) --> (52, 52, 128)
        print('layer7.shape', d['leaky_relu7'].get_shape().as_list())

        #conv8 - batch_norm8 - leaky_relu8 - pool8
        with tf.variable_scope('layer8'):
            d['conv8'] = conv_layer(d['leaky_relu7'], 3, 1, 256,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 52, 52, 128), out shape :(B, 52, 52, 256)
            d['batch_norm8'] = batchNormalization(d['conv8'], is_train)                  # input modify for activation layer
            d['leaky_relu8'] = tf.nn.leaky_relu(d['batch_norm8'], alpha=0.1)
            d['pool8'] = max_pool(d['leaky_relu8'], 2, 2, padding='SAME')                # in shape :(B, 52, 52, 256), out shape :(B, 26, 26, 256)
        # (52, 52, 128) --> (26, 26, 256)
        print('layer8.shape', d['pool8'].get_shape().as_list())

        #conv9 - batch_norm9 - leaky_relu9
        with tf.variable_scope('layer9'):
            d['conv9'] = conv_layer(d['pool8'], 3, 1, 512,
                                    padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 26, 26, 256), out shape :(B, 26, 26, 512)
            d['batch_norm9'] = batchNormalization(d['conv9'], is_train)                  # input modify for activation layer
            d['leaky_relu9'] = tf.nn.leaky_relu(d['batch_norm9'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)
        print('layer9.shape', d['leaky_relu9'].get_shape().as_list())

        #conv10 - batch_norm10 - leaky_relu10
        with tf.variable_scope('layer10'):
            d['conv10'] = conv_layer(d['leaky_relu9'], 1, 1, 256,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 26, 26, 512), out shape :(B, 26, 26, 256)
            d['batch_norm10'] = batchNormalization(d['conv10'], is_train)                 # input modify for activation layer
            d['leaky_relu10'] = tf.nn.leaky_relu(d['batch_norm10'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)
        print('layer10.shape', d['leaky_relu10'].get_shape().as_list())

        #conv11 - batch_norm11 - leaky_relu11
        with tf.variable_scope('layer11'):
            d['conv11'] = conv_layer(d['leaky_relu10'], 3, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 26, 26, 256), out shape :(B, 26, 26, 512)
            d['batch_norm11'] = batchNormalization(d['conv11'], is_train)                 # input modify for activation layer
            d['leaky_relu11'] = tf.nn.leaky_relu(d['batch_norm11'], alpha=0.1)
        # (26, 26, 256) --> (26, 26, 512)
        print('layer11.shape', d['leaky_relu11'].get_shape().as_list())

        #conv12 - batch_norm12 - leaky_relu12
        with tf.variable_scope('layer12'):
            d['conv12'] = conv_layer(d['leaky_relu11'], 1, 1, 256,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 26, 26, 512), out shape :(B, 26, 26, 256)
            d['batch_norm12'] = batchNormalization(d['conv12'], is_train)                 # input modify for activation layer
            d['leaky_relu12'] = tf.nn.leaky_relu(d['batch_norm12'], alpha=0.1)
        # (26, 26, 512) --> (26, 26, 256)
        print('layer12.shape', d['leaky_relu12'].get_shape().as_list())

        #conv13 - batch_norm13 - leaky_relu13 - pool13
        with tf.variable_scope('layer13'):
            d['conv13'] = conv_layer(d['leaky_relu12'], 3, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 26, 26, 256), out shape :(B, 26, 26, 512)
            d['batch_norm13'] = batchNormalization(d['conv13'], is_train)                 # input modify for activation layer
            d['leaky_relu13'] = tf.nn.leaky_relu(d['batch_norm13'], alpha=0.1)
            d['pool13'] = max_pool(d['leaky_relu13'], 2, 2, padding='SAME')               # in shape :(B, 26, 26, 512), out shape :(B, 13, 13, 512)
        # (26, 26, 256) --> (13, 13, 512)
        print('layer13.shape', d['pool13'].get_shape().as_list())

        #conv14 - batch_norm14 - leaky_relu14
        with tf.variable_scope('layer14'):
            d['conv14'] = conv_layer(d['pool13'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 512), out shape :(B, 13, 13, 1024)
            d['batch_norm14'] = batchNormalization(d['conv14'], is_train)                 # input modify for activation layer
            d['leaky_relu14'] = tf.nn.leaky_relu(d['batch_norm14'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer14.shape', d['leaky_relu14'].get_shape().as_list())

        #conv15 - batch_norm15 - leaky_relu15
        with tf.variable_scope('layer15'):
            d['conv15'] = conv_layer(d['leaky_relu14'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 1024), out shape :(B, 13, 13, 512)
            d['batch_norm15'] = batchNormalization(d['conv15'], is_train)                 # input modify for activation layer
            d['leaky_relu15'] = tf.nn.leaky_relu(d['batch_norm15'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)
        print('layer15.shape', d['leaky_relu15'].get_shape().as_list())

        #conv16 - batch_norm16 - leaky_relu16
        with tf.variable_scope('layer16'):
            d['conv16'] = conv_layer(d['leaky_relu15'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 512), out shape :(B, 13, 13, 1024)
            d['batch_norm16'] = batchNormalization(d['conv16'], is_train)                 # input modify for activation layer
            d['leaky_relu16'] = tf.nn.leaky_relu(d['batch_norm16'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer16.shape', d['leaky_relu16'].get_shape().as_list())

        #conv17 - batch_norm16 - leaky_relu17
        with tf.variable_scope('layer17'):
            d['conv17'] = conv_layer(d['leaky_relu16'], 1, 1, 512,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 1024), out shape :(B, 13, 13, 512)
            d['batch_norm17'] = batchNormalization(d['conv17'], is_train)
            d['leaky_relu17'] = tf.nn.leaky_relu(d['batch_norm17'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 512)
        print('layer17.shape', d['leaky_relu17'].get_shape().as_list())

        #conv18 - batch_norm18 - leaky_relu18
        with tf.variable_scope('layer18'):
            d['conv18'] = conv_layer(d['leaky_relu17'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 512), out shape :(B, 13, 13, 1024)
            d['batch_norm18'] = batchNormalization(d['conv18'], is_train)                 # input modify for activation layer
            d['leaky_relu18'] = tf.nn.leaky_relu(d['batch_norm18'], alpha=0.1)
        # (13, 13, 512) --> (13, 13, 1024)
        print('layer18.shape', d['leaky_relu18'].get_shape().as_list())

        #conv19 - batch_norm19 - leaky_relu19
        with tf.variable_scope('layer19'):
            d['conv19'] = conv_layer(d['leaky_relu18'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 1024), out shape :(B, 13, 13, 1024)
            d['batch_norm19'] = batchNormalization(d['conv19'], is_train)                 # input modify for activation layer
            d['leaky_relu19'] = tf.nn.leaky_relu(d['batch_norm19'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer19.shape', d['leaky_relu19'].get_shape().as_list())

        #conv20 - batch_norm20 - leaky_relu20
        with tf.variable_scope('layer20'):
            d['conv20'] = conv_layer(d['leaky_relu19'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 13, 13, 512), out shape :(B, 13, 13, 1024)
            d['batch_norm20'] = batchNormalization(d['conv20'], is_train)                 # input modify for activation layer
            d['leaky_relu20'] = tf.nn.leaky_relu(d['batch_norm20'], alpha=0.1)
        # (13, 13, 1024) --> (13, 13, 1024)
        print('layer20.shape', d['leaky_relu20'].get_shape().as_list())

        # concatenate layer20 and layer 13 using space to depth
        # skip_connection - skip_batch - skip_leaky_relu - skip_space_to_depth_x2 - concat21
        with tf.variable_scope('layer21'):
            d['skip_connection'] = conv_layer(d['leaky_relu13'], 1, 1, 64,
                                              padding='SAME', use_bias=False, weights_stddev=0.01) # in shape :(B, 26, 26, 512), out shape :(B, 26, 26, 64)
            d['skip_batch'] = batchNormalization(
                d['skip_connection'], is_train)                                                    # input modify for activation layer
            d['skip_leaky_relu'] = tf.nn.leaky_relu(d['skip_batch'], alpha=0.1)
            d['skip_space_to_depth_x2'] = tf.space_to_depth(
                d['skip_leaky_relu'], block_size=2)                                                # 풀링 비슷한 작업인데 4개의 이미지로 쪼개서 채널 만듦
                                                                                                   # in shape :(B, 26, 26, 64), out shape :(B, 13, 13, 256)
            d['concat21'] = tf.concat(
                [d['skip_space_to_depth_x2'], d['leaky_relu20']], axis=-1)                         # out shape :(B, 13, 13, 256+1024)
        # (13, 13, 1024) --> (13, 13, 256+1024)
        print('layer21.shape', d['concat21'].get_shape().as_list())

        #conv22 - batch_norm22 - leaky_relu22
        with tf.variable_scope('layer22'):
            d['conv22'] = conv_layer(d['concat21'], 3, 1, 1024,
                                     padding='SAME', use_bias=False, weights_stddev=0.01) # # in shape :(B, 13, 13, 1280), out shape :(B, 13, 13, 1024)
            d['batch_norm22'] = batchNormalization(d['conv22'], is_train)                 # input modify for activation layer
            d['leaky_relu22'] = tf.nn.leaky_relu(d['batch_norm22'], alpha=0.1)
        # (13, 13, 1280) --> (13, 13, 1024)
        print('layer22.shape', d['leaky_relu22'].get_shape().as_list())

        output_channel = self.num_anchors * (5 + self.num_classes)                                    # output_channel = 5 * (5 + 1) = 30
        d['logit'] = conv_layer(d['leaky_relu22'], 1, 1, output_channel,
                                padding='SAME', use_bias=True, weights_stddev=0.01, biases_value=0.1) # in shape :(B, 13, 13, 1024), out shape :(B, 13, 13, 30)
        d['pred'] = tf.reshape(
            d['logit'], (-1, self.grid_size[0], self.grid_size[1], self.num_anchors, 5 + self.num_classes)) # out shape :(B, 13, 13, 5, 6)
        print('pred.shape', d['pred'].get_shape().as_list())
        # (13, 13, 1024) --> (13, 13, num_anchors , (5 + num_classes))

        return d

    def _build_loss(self, **kwargs):
        """
        Build loss function for the model training.
        :param kwargs: dict, extra arguments
                - loss_weights: list, [xy, wh, resp_confidence, no_resp_confidence, class_probs]
        :return tf.Tensor.
        """

        loss_weights = kwargs.pop('loss_weights', [5, 5, 5, 0.5, 1.0]) # train.py에서 제공된 **kwargs변수가 없음, loss_weights = [5, 5, 5, 0.5, 1.0]
        # DEBUG
        # loss_weights = kwargs.pop('loss_weights', [1.0, 1.0, 1.0, 1.0, 1.0])
        grid_h, grid_w = self.grid_size                             # grid_size = [13, 13]
        num_classes = self.num_classes                              # num_classes = 1
        anchors = self.anchors                                      # anchors shape : (5, 2)
        grid_wh = np.reshape([grid_w, grid_h], [
                             1, 1, 1, 1, 2]).astype(np.float32)     # grud_wh = [[[[[13., 13.]]]]]
        cxcy = np.transpose([np.tile(np.arange(grid_w), grid_h),
                             np.repeat(np.arange(grid_h), grid_w)]) # 그리드의 중심좌표판 생성 cxcy shape : (169, 2)
        cxcy = np.reshape(cxcy, (1, grid_h, grid_w, 1, 2))          # 형상 변경 cxcy shape : (1, 13, 13, 1, 2)

        txty, twth = self.pred[..., 0:2], self.pred[..., 2:4]                                   # box 크기 정보 추출 shape :(B, 13, 13, 5, 2)
        confidence = tf.sigmoid(self.pred[..., 4:5])                                            # confi정보 추출 shape :(B, 13, 13, 5, 1)
        class_probs = tf.nn.softmax(
            self.pred[..., 5:], axis=-1) if num_classes > 1 else tf.sigmoid(self.pred[..., 5:]) # 여기서는 sigmoid사용 shape :(B, 13, 13, 5, 1)
        bxby = tf.sigmoid(txty) + cxcy                                                          # pred한 정확한 grid상의 bbox중심좌표 위치 생성
                                                                                                # 전체 grid에서의 정확한 좌표값이 pred되는게 아니라 
                                                                                                # 1grid안에서의 상대 위치값이 pred되게 만들어서 인듯
        pwph = np.reshape(anchors, (1, 1, 1, self.num_anchors, 2)) / 32                         # anchors를 그리드상의 길이정보로 바꾸고 형태를 reshape
        bwbh = tf.exp(twth) * pwph                                                              # pred한 정확한 grid상의 bbox크기 정보 생성

        # calculating for prediction
        nxny, nwnh = bxby / grid_wh, bwbh / grid_wh           # 1x1이미지 형태안의 bbox(위치, 크기 정보)모습으로 정규화,  shape :(B, 13, 13, 5, 2)
        nx1ny1, nx2ny2 = nxny - 0.5 * nwnh, nxny + 0.5 * nwnh # bbox의 중심좌표와 길이를 이용 bbox의 각 꼭지점 좌표 획득, shape :(B, 13, 13, 5, 2)
        self.pred_y = tf.concat(
            (nx1ny1, nx2ny2, confidence, class_probs), axis=-1)   # build된 model의 pred결과를 해석한 정보들을 pred_y에 저장, shape :(B, 13, 13, 5, 6)
                                                                  # valid set 을 pred하기 위해 필요해서 따로 pred_y에 저장(nn.py line 72)

        # calculating IoU for metric
        num_objects = tf.reduce_sum(self.y[..., 4:5], axis=[1, 2, 3, 4]) # self.y = y_true(batch) shape : (B, 13, 13, 5, 6), num_objects shape : (B, )
        max_nx1ny1 = tf.maximum(self.y[..., 0:2], nx1ny1)                # intersection을 구하기 위한 작업, max_nx1ny1 shape : (B, 13, 13, 5, 2)
        min_nx2ny2 = tf.minimum(self.y[..., 2:4], nx2ny2)                # intersection을 구하기 위한 작업, min_nx2ny2 shape : (B, 13, 13, 5, 2)
        intersect_wh = tf.maximum(min_nx2ny2 - max_nx1ny1, 0.0)          # intersection의 가로 세로 길이 , intersect_wh shape : (B, 13, 13, 5, 2)
        intersect_area = tf.reduce_prod(intersect_wh, axis=-1)           # intersection의 가로 세로 길이의 곱, intersect_area shape : (B, 13, 13, 5)
        intersect_area = tf.where(
            tf.equal(intersect_area, 0.0), tf.zeros_like(intersect_area), intersect_area) # 결과가 같은데 왜하는거지 ??
        gt_box_area = tf.reduce_prod(
            self.y[..., 2:4] - self.y[..., 0:2], axis=-1)                # True box area를 구함 gt_box_area shape : (B, 13, 13, 5)
        box_area = tf.reduce_prod(nx2ny2 - nx1ny1, axis=-1)              # pred box area를 구함 box_area shape : (B, 13, 13 ,5)
        iou = tf.truediv(
            intersect_area, (gt_box_area + box_area - intersect_area))   # iou값 구하기(intersection / union) iou shape : (B, 13, 13 ,5)
        sum_iou = tf.reduce_sum(iou, axis=[1, 2, 3])                     # 이미지별 object들의 iou의 합을 구합sum_iou shape : (B, )
        self.iou = tf.truediv(sum_iou, num_objects)                      # sum_iou / num_objects, shape : (B, )

        gt_bxby = 0.5 * (self.y[..., 0:2] + self.y[..., 2:4]) * grid_wh  # True bbox의 grid상 중심좌표 얻음(y에 좌표 0~1로 정규화 됨) shape :(B, 13, 13, 5, 2)
        gt_bwbh = (self.y[..., 2:4] - self.y[..., 0:2]) * grid_wh        # True bbox의 grid상 가로 세로 길이를 얻어옴 shape :(B, 13, 13, 5, 2)

        resp_mask = self.y[..., 4:5]                                     # 검출할 책임 있음(1), 없음(0)으로 된 mask 배열 shape :(B, 13, 13, 5, 1)
        no_resp_mask = 1.0 - resp_mask                                   # 검출할 책임 있음(0), 없음(1)으로 된 mask 배열 shape :(B, 13, 13, 5, 1)
        gt_confidence = resp_mask * tf.expand_dims(iou, axis=-1)         # mask와 각 현 batch내의 bbox의 iou를 곱해 confidence score를 구함(sum_iou아님) 
        gt_class_probs = self.y[..., 5:]                                 # gt_confidence, gt_class_probs shape : (B, 13, 13, 5, 1)

        loss_bxby = loss_weights[0] * resp_mask * \
            tf.square(gt_bxby - bxby)                          # bbox 중심 좌표에 관한 loss(가중치 5)            shape : (B, 13, 13, 5, 2)
        loss_bwbh = loss_weights[1] * resp_mask * \
            tf.square(tf.sqrt(gt_bwbh) - tf.sqrt(bwbh))        # bbox 가로 세로 길이에 관한 loss(가중치 5)       shape : (B, 13, 13, 5, 2)
        loss_resp_conf = loss_weights[2] * resp_mask * \
            tf.square(gt_confidence - confidence)              # True object confidence에 관한 loss(가중치 5)    shape : (B, 13, 13, 5, 1)
        loss_no_resp_conf = loss_weights[3] * no_resp_mask * \
            tf.square(gt_confidence - confidence)              # False object confidence에 관한 loss(가중치 0.5) shape : (B, 13, 13, 5, 1)
        loss_class_probs = loss_weights[4] * resp_mask * \
            tf.square(gt_class_probs - class_probs)            # class_prob에 관한 loss(가중치 1)                shape : (B, 13, 13, 5, 1)

        merged_loss = tf.concat((
                                loss_bxby,
                                loss_bwbh,
                                loss_resp_conf,
                                loss_no_resp_conf,
                                loss_class_probs
                                ),
                                axis=-1)                       # 모든 loss값 이어붙이기 shape : (B, 13, 13, 5, 7)
        #self.merged_loss = merged_loss
        total_loss = tf.reduce_sum(merged_loss, axis=-1)       # total_loss shape : (B, 13, 13, 5)
        total_loss = tf.reduce_mean(total_loss)                # total_loss shape : ()
        return total_loss                                      # 이미지별 total_loss를 반환하는 것이 아닌 배치별 total_loss를 반환

    # def interpret_output(self, sess, images, **kwargs):
    #     """
    #     Interpret outputs to decode bounding box from y_pred.
    #     :param sess: tf.Session
    #     :param kwargs: dict, extra arguments for prediction.
    #             -batch_size: int, batch size for each iteraction.
    #     :param images: np.ndarray, shape (N, H, W, C)
    #     :return bbox_pred: np.ndarray, shape (N, grid_h*grid_w*num_anchors, 5 + num_classes)
    #     """
    #     batch_size = kwargs.pop('batch_size', 32)
    #     is_batch = len(images.shape) == 4
    #     if not is_batch:
    #         images = np.expand_dims(images, 0)
    #     pred_size = images.shape[0]
    #     num_steps = pred_size // batch_size

    #     bboxes = []
    #     for i in range(num_steps + 1):
    #         if i == num_steps:
    #             image = images[i * batch_size:]
    #         else:
    #             image = images[i * batch_size:(i + 1) * batch_size]
    #         bbox = sess.run(self.pred_y, feed_dict={
    #                         self.X: image, self.is_train: False})
    #         bbox = np.reshape(bbox, (bbox.shape[0], -1, bbox.shape[-1]))
    #         bboxes.append(bbox)
    #     bboxes = np.concatenate(bboxes, axis=0)

    #     if is_batch:
    #         return bboxes
    #     else:
    #         return bboxes[0]
