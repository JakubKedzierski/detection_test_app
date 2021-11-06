import os
import sys
from object_detection.utils import ops as utils_ops
sys.path.append(os.path.abspath("./Mask_RCNN/"))
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.config import Config
from depth_analyze import DepthAnalyze, analyzeDepth
from base_detection import DetectionModule
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2 as cv
from logger import Logger
from Commons import ModelsStrings, ObjectDetectionType, Display

class OutputInfo:
    def __init__(self, masks, bboxes, scores, class_ids, categories = []):
        self.masks = masks
        self.bboxes = bboxes # format [yLD, xLD, yRU, xRU]
        self.scores = scores
        self.class_ids = class_ids
        self.categories = categories

class MaskRCNNModule(DetectionModule):

    def __init__(self, path_to_wages, detection_type):
        super().__init__()
        Logger.log(ModelsStrings.mask_rcnn, "Mask R-CNN model initialization")

        self.name = ""
        if detection_type is ObjectDetectionType.tomato:
            self.name = ObjectDetectionType.tomato
            self.num_classes = 3
            self.classes = ["background", "tomato", "unripe tomato"]
        else:
            self.name = ObjectDetectionType.apple
            self.num_classes = 2
            self.classes = ["background", "apple"]

        class InferenceConfig(Config):
            NAME = self.name.name
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            NUM_CLASSES = self.num_classes
            IMAGE_MIN_DIM = 720
            IMAGE_MAX_DIM = 1280
            BACKBONE = 'resnet101'
            RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)
            TRAIN_ROIS_PER_IMAGE = 32
            MAX_GT_INSTANCES = 50
            POST_NMS_ROIS_INFERENCE = 500
            POST_NMS_ROIS_TRAINING = 1000

        inference_config = InferenceConfig()

        self.model = modellib.MaskRCNN(mode="inference",
                                       config=inference_config,
                                       model_dir='')

        print("Loading weights from ", path_to_wages)
        self.model.load_weights(path_to_wages, by_name=True)

    def detect(self, frame, plot, image_depth=[], analyze_depth=False, display=Display.plt):

        start_time = time.time()
        results = self.model.detect([frame], verbose=1)
        end_time = time.time()
        print("Frame detection time: {:10.2f}s".format(end_time - start_time))

        r = results[0]

        output = OutputInfo(r['masks'], r['rois'], r['scores'], r['class_ids'],
                            self.classes)


        if analyze_depth:
            output = analyzeDepth(image_depth, output)

        plt.cla()
        visualize.display_instances(frame, output.bboxes, output.masks, output.class_ids,
                                    output.categories, output.scores, ax=plot, show_mask=False)


class YoloV3Module(DetectionModule):
    def __init__(self, classes_path, config_path, weight_path):
        super().__init__()
        self.c_threshold = 0.5  # set threshold for bounding box values
        self.nms = 0.4  # set threshold for non maximum supression
        self.width = 416  # width of input image
        self.height = 416  # height of input image

        self.classesFile = classes_path
        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        # initialize a list of colors to represent each possible class label
        self.COLORS = [(0, 0, 255), (0, 51, 25)]

        # PATH to weight and config files
        self.config = config_path
        self.weight = weight_path

        # Read the model using dnn
        self.net = cv.dnn.readNetFromDarknet(self.config, self.weight)

    def detect(self, frame, plot, image_depth=[], analyze_depth=False, display=Display.plt):
        image = frame
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Get the names of output layers
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # generate blob for image input to the network
        blob = cv.dnn.blobFromImage(image, 1 / 255, (416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)

        start = time.time()
        layersOutputs = self.net.forward(ln)
        end = time.time()
        print("Detection time: {:1.2f}s".format(end - start))

        boxes, confidences, classIDs = self.convert_output(image, layersOutputs)
        masks = []
        boxes = self.convert_bbox_to_output(boxes)
        output = OutputInfo(masks, np.array(boxes), np.array(confidences), np.array(classIDs))
        copy_out = output

        if analyze_depth:
            output = analyzeDepth(image_depth, output)
        print("----------------")
        output = self.convert_output_to_display(output)

        self.display(output.bboxes.tolist(), output.scores, output.class_ids, image, plot, display=display)

    def convert_bbox_to_output(self, bboxes):

        for i in range (0, len(bboxes)):
            bbox = bboxes[i].copy()
            bboxes[i][0] = bbox[1]
            bboxes[i][1] = bbox[0]
            bboxes[i][2] = bbox[1] + bbox[3]
            bboxes[i][3] = bbox[0] + bbox[2]

            if bboxes[i][0] > 720:
                bboxes[i][0] = 720
            if bboxes[i][1] > 1280:
                bboxes[i][1] = 1280
            if bboxes[i][2] > 720:
                bboxes[i][2] = 720
            if bboxes[i][3] > 1280:
                bboxes[i][3] = 1280

            if bboxes[i][0] < 0:
                bboxes[i][0] = 0
            if bboxes[i][1] < 0:
                bboxes[i][1] = 0
            if bboxes[i][2] < 0:
                bboxes[i][2] = 0
            if bboxes[i][3] < 0:
                bboxes[i][3] = 0

        return bboxes

    def convert_output_to_display(self, output):
        # format [yLD, xLD, yRU, xRU] mask rcnn + output
        # format [xLD, yLD, width, height] yolo
        # format [start_point, end_point] cv rectangle:
        for i in range(0, len(output.bboxes)):
            bbox = output.bboxes[i].copy()
            output.bboxes[i][0] = bbox[1]
            output.bboxes[i][1] = bbox[0]
            output.bboxes[i][2] = bbox[3] - bbox[1]
            output.bboxes[i][3] = bbox[2] - bbox[0]

        return output

    def convert_output(self, image, layersOutputs):
        (H, W) = image.shape[:2]
        boxes = []
        confidences = []
        classIDs = []

        for output in layersOutputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.c_threshold:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    classIDs.append(classID)
        return boxes, confidences, classIDs

    def display(self, boxes, confidences, classIDs, image, plot, display = Display.plt):

        # Remove unnecessary boxes using non maximum suppression
        idxs = cv.dnn.NMSBoxes(boxes, confidences, self.c_threshold, self.nms)

        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.classes[classIDs[i]], confidences[i])
                cv.putText(image, text, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX,
                           0.6, (255, 255, 255), 2)
        if display is Display.plt:
            image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            plot.imshow(image)
        else:
            cv.imshow("Detection result", image)
            cv.waitKey(100)

        return image

import tensorflow as tf

class SSDModule(DetectionModule):
    def __init__(self, pb_file_path, detection_type=ObjectDetectionType.tomato):
        self.detection_type = detection_type

        if detection_type is ObjectDetectionType.tomato:
            self.category_index = {1: {'id': 1, 'name': 'greenfruit'}, 2: {'id': 2, 'name': 'redfruit'}}
        elif detection_type is ObjectDetectionType.apple:
            self.category_index = {1: {'id': 1, 'name': 'apple'}}

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(pb_file_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def fast_detect(self, pipeline,align, analyze_depth=False):
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                while True:

                    frames = pipeline.wait_for_frames(timeout_ms=200)
                    if frames.size() < 2:
                        continue
                    aligned_frames = align.process(frames)
                    image_depth = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()
                    image = np.asanyarray(color_frame.get_data())

                    plt.pause(0.1)
                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()

                    all_tensor_names = {
                        output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                tensor_name)
                    if 'detection_masks' in tensor_dict:
                        # The following processing is only for single image
                        detection_boxes = tf.squeeze(
                            tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(
                            tensor_dict['detection_masks'], [0])
                        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                        real_num_detection = tf.cast(
                            tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [
                            real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                            real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, image.shape[0], image.shape[1])
                        detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        # Follow the convention by adding back the batch dimension
                        tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                    bboxes = self.convert_bboxes_to_output(output_dict['detection_boxes'], output_dict['detection_scores'],
                                                           0.5)
                    # format xLD, yLD, xRU, yRU ssd
                    masks = []
                    class_ids = []
                    output = OutputInfo(masks, np.array(bboxes), np.array(output_dict['detection_scores']), class_ids)
                    if analyze_depth:
                        output = analyzeDepth(image_depth, output)
                    output_dict['detection_boxes'] = np.array(
                        self.convert_output_to_ssd_vizualisation_bboxes(output.bboxes))
                    self.display(output_dict, image)

    def detect(self, image, plot, image_depth=[], analyze_depth=False, display=Display.plt):
        output_dict = []
        with self.detection_graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()

                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                        real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                        real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]

                bboxes = self.convert_bboxes_to_output(output_dict['detection_boxes'], output_dict['detection_scores'],
                                                       0.5)
                # format xLD, yLD, xRU, yRU ssd
                masks = []
                class_ids = []
                output = OutputInfo(masks, np.array(bboxes), np.array(output_dict['detection_scores']), class_ids)
                if analyze_depth:
                    output = analyzeDepth(image_depth, output)
                output_dict['detection_boxes'] = np.array(
                    self.convert_output_to_ssd_vizualisation_bboxes(output.bboxes))
                self.display(output_dict, image)

    def convert_bboxes_to_output(self, bboxes, scores, thresold):
        boxes = np.squeeze(bboxes)
        scores = np.squeeze(scores)
        min_score_thresh = 0.5
        bboxes = boxes[scores > min_score_thresh]
        im_width = 1280
        im_height = 720
        final_box = []
        for box in bboxes:
            ymin, xmin, ymax, xmax = box
            # format xymin, xmin, ymax, xmax  ssd - normalizowane
            final_box.append([ymin * im_height, xmin * im_width, ymax * im_height, xmax * im_width ])
            #format yLD, xLD, yRU, xRU ssd

        return final_box

    def convert_output_to_ssd_vizualisation_bboxes(self, bboxes):
        im_width = 1280
        im_height = 720
        final_box = []
        for box in bboxes:
            y0, x0, y1, x1 = box
            final_box.append([y0 / im_height, x0 / im_width, y1 / im_height, x1 / im_width])

        return final_box


    def display(self, output_dict, image, display=Display.plt):

        if 'detection_masks' in output_dict:
            output_dict['detection_masks'] = output_dict['detection_masks'][0]


        from object_detection.utils import visualization_utils as vis_util
        import matplotlib
        matplotlib.use('Qt5Agg')
        vis_util.visualize_boxes_and_labels_on_image_array(
                image,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks'),
                use_normalized_coordinates=True,
                line_thickness=6,
                min_score_thresh=0.5)
        cv.imshow('Test video', cv.cvtColor(image, cv.COLOR_BGR2RGB))

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            exit(0)
