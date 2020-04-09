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
            box = [xmin, ymin, xmax, ymax]
            final_boxes.append(box)
        return final_boxes

    def predict(self, file):
        """
        Predict objects on specified image.

        :param file: file-like object, string
        :return: List of bounding boxes (as a list) that store result,
                 [[xmin, ymin, xmax, ymax], ..., [...]]
        """

        np_img = np.fromfile(file, np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

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

    def _post_process(self, images, detections):
        """
        Process detection results based on confidence threshold level.
        """
        # matrix result
        boxes = [[] for _ in range(len(images))]
        for i in range(0, detections.shape[2]):
            img_index = int(detections[0, 0, i, 0])  # fetch image from batch
            confidence = detections[0, 0, i, 2]
            if confidence > self._conf_threshold:
                (h, w) = images[img_index].shape[:2]
                np_box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                boxes[img_index].append(np_box.astype("int").tolist())
        return boxes

    def predict(self, files):
        """
        Predict objects on specified images.

        :param files: A list/tuple of file-like objects, strings
        :return: List of bounding boxes list that stores detection results
                 per image, [[[xmin, ymin, xmax, ymax]]], e.g.:
                 [
                   [[xmin, ymin, xmax, ymax]],  # <- 1
                   [],  # <- No detections for second image
                   [[xmin, ymin, xmax, ymax], [xmin, ymin, xmax, ymax]]  # <- 2
                 ]
        """

        if not isinstance(files, (list, tuple)):
            files = [files]

        np_images = [np.fromfile(file, np.uint8) for file in files]
        images = [cv2.imdecode(np_img, cv2.IMREAD_COLOR)
                  for np_img in np_images]

        # TODO: There is a problem with prediction results if `images` list
        #  size changed 'dynamically'.
        blob = cv2.dnn.blobFromImages(
            images,
            1.0,
            (self.image_width, self.image_height),
            (104, 177, 123),
            1,
            crop=False
        )

        # Sets the input to the network
        self.net.setInput(blob)

        # Runs the forward pass to get output of the output layers
        detections = self.net.forward()

        # Remove the bounding boxes with low confidence
        objects = self._post_process(images, detections)

        return objects
