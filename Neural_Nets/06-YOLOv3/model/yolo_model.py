"""YOLO v3 Model Implementation - Fixed for TensorFlow 2.x"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K


class YOLO:
    def __init__(self, obj_threshold, nms_threshold):
        """Initialize YOLO model.

        # Arguments:
            obj_threshold: Float, confidence threshold for object detection.
            nms_threshold: Float, non-max suppression threshold.
        """
        self._t1 = obj_threshold
        self._t2 = nms_threshold
        self._yolo = load_model('data/yolo.h5', compile=False)  # Ensure proper model loading

    def _process_feats(self, out, anchors, mask):
        """Process YOLO output features.

        # Arguments:
            out: Tensor, output feature map of YOLO.
            anchors: List, anchor box dimensions.
            mask: List, mask for anchors.

        # Returns:
            boxes, box_confidence, box_class_probs
        """
        grid_h, grid_w, num_boxes = out.shape[1:4]
        anchors = [anchors[i] for i in mask]

        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, len(anchors), 2])

        out = out[0]  # Remove batch dimension
        box_xy = tf.sigmoid(out[..., :2]).numpy()  # Convert to NumPy array
        box_wh = (tf.exp(out[..., 2:4]) * anchors_tensor).numpy()
        box_confidence = tf.sigmoid(out[..., 4]).numpy().reshape(grid_h, grid_w, num_boxes, 1)
        box_class_probs = tf.sigmoid(out[..., 5:]).numpy()

        col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
        row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

        col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
        grid = np.concatenate((col, row), axis=-1)

        box_xy += grid
        box_xy /= (grid_w, grid_h)
        box_wh /= (416, 416)
        box_xy -= (box_wh / 2.)
        boxes = np.concatenate((box_xy, box_wh), axis=-1)

        return boxes, box_confidence, box_class_probs

    def _filter_boxes(self, boxes, box_confidences, box_class_probs):
        """Filter boxes using the object threshold."""
        box_scores = box_confidences * box_class_probs
        box_classes = np.argmax(box_scores, axis=-1)
        box_class_scores = np.max(box_scores, axis=-1)
        pos = np.where(box_class_scores >= self._t1)

        return boxes[pos], box_classes[pos], box_class_scores[pos]

    def _nms_boxes(self, boxes, scores):
        """Apply Non-Maximum Suppression (NMS)"""
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = w * h
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1)
            h1 = np.maximum(0.0, yy2 - yy1)
            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            order = order[np.where(ovr <= self._t2)[0] + 1]

        return np.array(keep)

    def _yolo_out(self, outs, shape):
        """Process YOLO output."""
        masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

        boxes, classes, scores = [], [], []

        for out, mask in zip(outs, masks):
            b, c, s = self._process_feats(out, anchors, mask)
            b, c, s = self._filter_boxes(b, c, s)
            boxes.append(b)
            classes.append(c)
            scores.append(s)

        if not boxes:
            return None, None, None

        boxes = np.concatenate(boxes)
        classes = np.concatenate(classes)
        scores = np.concatenate(scores)

        width, height = shape[1], shape[0]
        image_dims = np.array([width, height, width, height])
        boxes *= image_dims

        nboxes, nclasses, nscores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b, c, s = boxes[inds], classes[inds], scores[inds]

            keep = self._nms_boxes(b, s)
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

        if not nclasses and not nscores:
            return None, None, None

        return np.concatenate(nboxes), np.concatenate(nclasses), np.concatenate(nscores)

    def predict(self, image, shape):
        """Detect objects in the image using YOLO."""
        image = np.expand_dims(image, axis=0)  # Ensure correct input shape
        outs = self._yolo.predict(image)
        return self._yolo_out(outs, shape)
