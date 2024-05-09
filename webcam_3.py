import numpy as np
import cv2
import tensorflow as tf
import pytesseract
import pyttsx3
import os
import urllib
import tarfile

MODEL_NAME = 'faster_rcnn_resnet50_coco_2018_01_28'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

if not os.path.exists(MODEL_NAME):
    print('Downloading the model')
    opener = urllib.request.URLopener()
    opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
    tar_file = tarfile.open(MODEL_FILE)
    for file in tar_file.getmembers():
        file_name = os.path.basename(file.name)
        if 'frozen_inference_graph.pb' in file_name:
            tar_file.extract(file, os.getcwd())
    print('Download complete')
else:
    print('Model already exists')

# Initialize TensorFlow object detection model
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = {1: {'id': 1, 'name': 'person'}, 2: {'id': 2, 'name': 'bicycle'}, 3: {'id': 3, 'name': 'car'}, 6: {'id': 6, 'name': 'bus'}, 8: {'id': 8, 'name': 'truck'}, 44: {'id': 44, 'name': 'bottle'}, 73: {'id': 73, 'name': 'laptop'}, 77: {'id': 77, 'name': 'cell phone'}, 84: {'id': 84, 'name': 'book'}}
category_index = label_map

# Initialize TTS engine
engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Use GPU for TensorFlow operations
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
with detection_graph.as_default(), tf.compat.v1.Session(graph=detection_graph, config=config) as sess:
    while True:
        ret, image_np = cap.read()

        # Perform object detection
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})

        # Visualize and announce distance of detected objects
        for i, b in enumerate(boxes[0]):
            if scores[0][i] >= 0.7:
                class_id = int(classes[0][i])
                if class_id in label_map:
                    class_name = label_map[class_id]['name']
                    
                    mid_x = (boxes[0][i][1] + boxes[0][i][3]) / 2
                    mid_y = (boxes[0][i][0] + boxes[0][i][2]) / 2
                    apx_distance = round(((1 - (boxes[0][i][3] - boxes[0][i][1])) ** 4), 1)
                    
                    cv2.putText(image_np, '{}'.format(apx_distance), (int(mid_x * 800), int(mid_y * 450)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                    if apx_distance <= 1 and 0.3 < mid_x < 0.7:
                        cv2.putText(image_np, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                        engine.say(f"Warning - {class_name} very close to the frame")
                        engine.runAndWait()

                    # Draw bounding box
                    (x, y, w, h) = (boxes[0][i][1] * 800, boxes[0][i][0] * 450, (boxes[0][i][3] - boxes[0][i][1]) * 800, (boxes[0][i][2] - boxes[0][i][0]) * 450)
                    cv2.rectangle(image_np, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
                    cv2.putText(image_np, class_name, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
                else:
                    print(f"Unknown class ID: {class_id}")


        cv2.imshow('Object Detection', cv2.resize(image_np, (1024, 768)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()