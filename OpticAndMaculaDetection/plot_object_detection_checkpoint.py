#!/usr/bin/env python
# coding: utf-8
"""
Object Detection From TF2 Checkpoint
====================================
"""

# %%
# This demo will take you through the steps of running an "out-of-the-box" TensorFlow 2 compatible
# detection model on a collection of images. More specifically, in this example we will be using
# the `Checkpoint Format <https://www.tensorflow.org/guide/checkpoint>`__ to load the model.

# %%
# Download the test images
# ~~~~~~~~~~~~~~~~~~~~~~~~
# First we will download the images that we will use throughout this tutorial. The code snippet
# shown bellow will download the test images from the `TensorFlow Model Garden <https://github.com/tensorflow/models/tree/master/research/object_detection/test_images>`_
# and save them inside the ``data/images`` folder.
import os
import random
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import numpy as np
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def check_folder(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir
# IMAGE_PATHS = download_images()
IMAGE_PATHS = []
NAMES = []
img_path = '/home/pham/Desktop/UFI/20210907/classification/abnormal/'
save_path = '/home/pham/Desktop/UFI/20210907/classification/abnormal_crop/'
check_folder(save_path)
for f in  sorted(os.listdir(img_path)):
    ext = os.path.splitext(f)[1]
    if ext == '.jpg':
        IMAGE_PATHS.append(img_path+f)
        NAMES.append(f)
print('number of images: ', len(NAMES))
PATH_TO_MODEL_DIR = "/home/pham/Documents/Python/TensorFlow/workspace/training_demo"
# %%

PATH_TO_LABELS = '/home/pham/Documents/Python/TensorFlow/workspace/training_demo/annotations/label_map.pbtxt'
# %%
# Load the model
# ~~~~~~~~~~~~~~
# Next we load the downloaded model
import time
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

PATH_TO_CFG = PATH_TO_MODEL_DIR + "/pipeline.config"
PATH_TO_CKPT = "/home/pham/Documents/Python/TensorFlow/workspace/training_demo/models/resnet_640_640"

print('Loading model... ')
start_time = time.time()

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-41')).expect_partial()

@tf.function
def detect_fn(image):
    """Detect objects in image."""

    image, shapes = detection_model.preprocess(image)
    # print(image.shape)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections, prediction_dict["feature_maps"]

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# %%
# Load label map data (for plotting)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Label maps correspond index numbers to category names, so that when our convolution network
# predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
# functions, but anything that returns a dictionary mapping integers to appropriate string labels
# would be fine.

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

# %%
# Putting everything together
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The code shown below loads an image, runs it through the detection model and visualizes the
# detection results, including the keypoints.
#
# Note that this will take a long time (several minutes) the first time you run this code due to
# tf.function's trace-compilation --- on subsequent runs (e.g. on new images), things will be
# faster.
#
# Here are some simple things to try out if you are curious:
#
# * Modify some of the input images and see if detection still works. Some simple things to try out here (just uncomment the relevant portions of code) include flipping the image horizontally, or converting to grayscale (note that we still expect the input image to have 3 channels).
# * Print out `detections['detection_boxes']` and try to match the box locations to the boxes in the image.  Notice that coordinates are given in normalized form (i.e., in the interval [0, 1]).
# * Set ``min_score_thresh`` to other values (between 0 and 1) to allow more detections in or to filter out more detections.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.

    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.

    Args:
      path: the file path to the image

    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))
import math
def crop_ufi(ufi, optic_disc_box, fovea_box):
    height = ufi.shape[0]
    width = ufi.shape[1]
    optic_disc_box[0] *= height
    optic_disc_box[1] *= width
    optic_disc_box[2] *= height
    optic_disc_box[3] *= width
    fovea_box[0] *= height
    fovea_box[1] *= width
    fovea_box[2] *= height
    fovea_box[3] *= width
    optic_center = [(optic_disc_box[1]+optic_disc_box[3])//2, (optic_disc_box[0]+optic_disc_box[2])//2] # x y
    fovea = [(fovea_box[1]+fovea_box[3])//2, (fovea_box[0]+fovea_box[2])//2] # x y
    optic_radius = math.sqrt((optic_disc_box[0]-optic_center[1])**2 + (optic_disc_box[1]-optic_center[0])**2)

    center_x = (fovea[0] + optic_center[0]) // 2
    center_y = (fovea[1] + optic_center[1]) // 2

    low_x = center_x 
    high_x = fovea[0]

    if low_x > high_x:
        low_x = fovea[0]
        high_x = center_x

    # random center 
    image_center_x = random.randint(low_x, high_x)
    image_center_y = fovea[1] - (fovea[0]-image_center_x)*(fovea[1]-center_y)/(fovea[0]-center_x)

    # # center at fovea
    # image_center_x = fovea[0]
    # image_center_y = fovea[1]

    alpha = 2.0
    crop_radius = alpha*optic_radius + math.sqrt((optic_center[0]-image_center_x)**2 + (optic_center[1]-image_center_y)**2)
    
    xmin = int(image_center_x - crop_radius)
    xmax = int(image_center_x + crop_radius)
    ymin = int(image_center_y - crop_radius)
    ymax = int(image_center_y + crop_radius)
    return ufi[ymin:ymax, xmin:xmax, :]
    


i=0
import time
begin = time.time()
for image_path, fn in zip(IMAGE_PATHS, NAMES):
    # if os.path.exists(save_path+fn):
    #     continue
    # print(i)
    print('Running inference for {}... '.format(image_path))
    if i%100==0:
        print('processing image ', i)

    image_np = load_image_into_numpy_array(image_path)

    # Things to try:
    # Flip horizontally
    # image_np = np.fliplr(image_np).copy()

    # Convert image to grayscale
    # image_np = np.tile(
    #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)

    detections, feature_maps = detect_fn(input_tensor)
    map0 = feature_maps[0]
    map0 = np.asarray(map0)
    map0 = np.squeeze(map0)
    print('feature_maps ', map0.shape)
    print('max ', np.max(map0))
    print('min ', np.min(map0))
    for i in range(10):
        m = map0[:,:,i]
        m = m/np.max(m)*255
        m = m.astype(np.uint8)
        h,w = m.shape
        # vis2 = cv.CreateMat(h, w, cv2.CV_8UC1)
        m = cv2.applyColorMap(m, cv2.COLORMAP_TURBO)
        cv2.imwrite(save_path+str(i)+'.png', m)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    # print(detections)
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    img = image_np.copy()
    boxes = detections['detection_boxes']
    # print(type(boxes))
    classes = detections['detection_classes']+label_id_offset
    scores = detections['detection_scores']
    class_list = np.unique(classes)
    box_final = list()
    score_final = list()
    for cl in class_list:
        idx = np.where(classes==cl)
        sc = scores[idx]
        box = boxes[idx]
        a = np.argmax(sc)
        box_final.append(box[a])
        score_final.append(sc[a])
    box_final = np.asarray(box_final) # ymin xmin ymax xmax
    score_final = np.asarray(score_final)
    # print('class ', class_list)
    # print('boxes ',box_final)
    ## cropping
    if len(box_final) == 2: 
        optic_disc_box = box_final[0]
        fovea_box = box_final[1]
        
        cropped_img = crop_ufi(img, optic_disc_box,  fovea_box)
        tf.keras.preprocessing.image.save_img(save_path+fn, cropped_img)
    else:
        print('eeeeeeeeeeeeeeeeeeeeee')
        

    # viz_utils.visualize_boxes_and_labels_on_image_array(
    #         img,
    #         box_final,
    #         class_list,
    #         score_final,
    #         # detections['detection_boxes'],
    #         # detections['detection_classes']+label_id_offset,
    #         # detections['detection_scores'],
    #         category_index,
    #         use_normalized_coordinates=True,
    #         max_boxes_to_draw=200,
    #         min_score_thresh=.001,
    #         agnostic_mode=False)
    # plt.figure()
    # plt.imshow(img)
    # print('Done')
    # plt.savefig('/home/pham/Desktop/UFI/20210907/fail_result/'+fn)


    i = i+1
end = time.time()
print('average time: ', (end-begin)/len(NAMES))
# sphinx_gallery_thumbnail_number = 2