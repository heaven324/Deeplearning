from abc import abstractmethod, abstractproperty, ABCMeta
import numpy as np
from learning.utils import convert_boxes, predict_nms_boxes, cal_recall

class Evaluator(metaclass=ABCMeta):
    """Base class for evaluation functions."""

    @abstractproperty
    def worst_score(self):
        """
        The worst performance score.
        :return float.
        """
        pass
    @abstractproperty
    def mode(self):
        """
        the mode for performance score, either 'max' or 'min'
        e.g. 'max' for accuracy, AUC, precision and recall,
              and 'min' for error rate, FNR and FPR.
        :return: str.
        """
        pass

    @abstractmethod
    def score(self, y_true, y_pred):
        """
        Performance metric for a given prediction.
        This should be implemented.
        :param y_true: np.ndarray, shape: (N, 5 + num_classes).
        :param y_pred: np.ndarray, shape: (N, 5 + num_classes).
        :return float.
        """
        pass

    @abstractmethod
    def is_better(self, curr, best, **kwargs):
        """
        Function to return whether current performance score is better than current best.
        This should be implemented.
        :param curr: float, current performance to be evaluated.
        :param best: float, current best performance.
        :return bool.
        """
        pass

class RecallEvaluator(Evaluator):
    """ Evaluator with Recall metric."""  # 위치를 찾고 올바르게 분류한 오브젝트 수 / 전체 오브젝트 수  

    @property
    def worst_score(self):
        """The worst performance score."""
        return 0.0

    @property
    def mode(self):
        """The mode for performance score."""
        return 'max'

    def score(self, y_true, y_pred, **kwargs):
        """Compute Recall for a given predicted bboxes"""
        nms_flag = kwargs.pop('nms_flag', True) # pop(키, 기본값) dict함수, 키 없을 시 기본값 반환
        if nms_flag:                            # nms_flag : Bool
            bboxes = predict_nms_boxes(y_pred)  # from learning.utils, y_pred shape: (N, g_H, g_W, anchors, 5 + num_classes)
        else:
            bboxes = convert_boxes(y_pred)      # from learning.utils
        gt_bboxes = convert_boxes(y_true)       # from learning.utils, y_true shape: (N, g_H, g_W, anchors, 5 + num_classes)
        score = cal_recall(gt_bboxes, bboxes)   # from learning.utils
        return score                            # score : float

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance scores is better than current best,
        with consideration of the relative threshold to the given performance score.
        :param kwargs: dict, extra arguments.
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4) # 0.0001
        relative_eps = 1.0 + score_threshold                  # 1.0001
        return curr > best * relative_eps                     # bool값을 return


class IoUEvaluator(Evaluator):
    """Evaluator with IoU(graph) metric."""

    @property
    def worst_score(self):
        """The worst performance score."""
        return 0.0

    @property
    def mode(self):
        """The mode for performance score."""
        return 'max'

    def score(self, sess, model, X, y):
        """
        Compute iou for a given prediction using YOLO model.
        :param sess: tf.Session.
        :param X: np.ndarray, sample image batches
        :param y: np.ndarray, sample labels batches
        :return iou: float. intersection of union
        """
        iou = sess.run(model.iou, feed_dict={model.X: X, model.y: y, model.is_train: False})
        score = np.mean(iou)
        return score

    def is_better(self, curr, best, **kwargs):
        """
        Return whether current performance scores is better than current best,
        with consideration of the relative threshold to the given performance score.
        :param kwargs: dict, extra arguments.
            - score_threshold: float, relative threshold for measuring the new optimum,
                               to only focus on significant changes.
        """
        score_threshold = kwargs.pop('score_threshold', 1e-4)
        relative_eps = 1.0 + score_threshold
        return curr > best * relative_eps