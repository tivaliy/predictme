import cv2
import numpy as np

from ._utils import to_point_form


class DarknetObjectDetector:
    """
    Class to predict objects using a Darknet Architecture with OpenCV.
    """

    img_width = 416
    image_height = 416

    def __init__(
            self,
            model_cfg,
            model_weights,
            conf_threshold=0.5,
            nms_threshold=0.4
    ):
        self._conf_threshold = conf_threshold
        self._nms_threshold = nms_threshold
        self.net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    @property
    def conf_threshold(self):
        """
        A threshold used to filter boxes by score.
        """
        return self._conf_threshold

    @property
    def nms_threshold(self):
        """
        A threshold used in non maximum suppression.
        """
        return self._nms_threshold

    def _get_outputs_names(self):
        """
        Get the names of the output layers.
        """

        # Get the names of all the layers in the network
        layers = self.net.getLayerNames()

        # Get the names of the output layers,
        # i.e. the layers with unconnected outputs
        return [layers[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

    def _post_process(self, image, outs):
        image_height = image.shape[0]
        image_width = image.shape[1]

        # Scan through all the bounding boxes output from the network and keep
        # only the ones with high confidence scores. Assign the box's class
        # label as the class with the highest score.
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * image_width)
                    center_y = int(detection[1] * image_height)
                    width = int(detection[2] * image_width)
                    height = int(detection[3] * image_height)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        # Remove redundant overlapping boxes with lower confidences
        indices = cv2.dnn.NMSBoxes(
            boxes,
            confidences,
            self.conf_threshold,
            self.nms_threshold
        )

        final_boxes = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left, top, width, height = box[0], box[1], box[2], box[3]
            xmin, ymin, xmax, ymax = to_point_form(left, top, width, height)
            box = [[xmin, ymin], [xmax, ymax]]
            final_boxes.append(box)
        return final_boxes

    def predict(self, image_path):
        """
        Predict objects on specified image.

        :param image_path: Path to image file.
        :return: List of bounding boxes (as a list) that store result,
                 [[xmin, ymin, xmax, ymax], [...], ..., [...]]
        """

        image = cv2.imread(image_path)

        # Create a 4D blob from a image.
        blob = cv2.dnn.blobFromImage(
            image,
            1 / 255,
            (self.img_width, self.image_height),
            (0, 0, 0),
            1,
            crop=False
        )

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward(self._get_outputs_names())

        # Remove the bounding boxes with low confidence
        objects = self._post_process(image, outs)

        return objects


class CaffeObjectDetector:
    """
    Class to predict objects using a Caffe Classification Model with OpenCV.
    """

    image_width = 300
    image_height = 300

    def __init__(
            self,
            proto_txt,
            model_weights,
            conf_threshold=0.5,
    ):
        self._conf_threshold = conf_threshold
        self.net = cv2.dnn.readNetFromCaffe(proto_txt, model_weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def _post_process(self, image, outs):
        (h, w) = image.shape[:2]
        boxes = []
        for i in range(0, outs.shape[2]):
            box = outs[0, 0, i, 3:7] * np.array([w, h, w, h])
            confidence = outs[0, 0, i, 2]
            if confidence > self._conf_threshold:
                boxes.append([[box[0], box[1]], [box[2], box[3]]])
        return boxes

    def predict(self, image_path):
        """
        Predict objects on specified image.

        :param image_path: Path to image file.
        :return: List of bounding boxes (as a list) that store objects result,
                 [[xmin, ymin, xmax, ymax], [...], ..., [...]]
        """

        image = cv2.imread(image_path)

        # Create a 4D blob from a image.
        blob = cv2.dnn.blobFromImage(
            image,
            1.0,
            (self.image_width, self.image_height),
            (104, 177, 123),
            1,
            crop=False
        )

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self.net.forward()

        # Remove the bounding boxes with low confidence
        objects = self._post_process(image, outs)

        return objects
