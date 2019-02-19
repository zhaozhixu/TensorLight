import tensorflow as tf
import numpy as np
import sys

def reorg_layer(feature_map, anchors, img_size, class_num):
    '''
    feature_map: a feature_map from [feature_map_1, feature_map_2, feature_map_3] returned
        from `forward` function
    anchors: shape: [3, 2]
    '''
    # NOTE: size in [h, w] format! don't get messed up!
    grid_size = feature_map.shape.as_list()[1:3]  # [13, 13]
    # the downscale ratio in height and weight
    ratio = tf.cast(img_size / grid_size, tf.float32)
    # rescale the anchors to the feature_map
    # NOTE: the anchor is in [w, h] format!
    rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]

    feature_map = tf.reshape(feature_map, [-1, grid_size[0], grid_size[1], 3, 5 + class_num])

    # split the feature_map along the last dimension
    # shape info: take 416x416 input image and the 13*13 feature_map for example:
    # box_centers: [N, 13, 13, 3, 2] last_dimension: [center_x, center_y]
    # box_sizes: [N, 13, 13, 3, 2] last_dimension: [width, height]
    # conf_logits: [N, 13, 13, 3, 1]
    # prob_logits: [N, 13, 13, 3, class_num]
    box_centers, box_sizes, conf_logits, prob_logits = tf.split(feature_map, [2, 2, 1, class_num], axis=-1)
    box_centers = tf.nn.sigmoid(box_centers)

    # use some broadcast tricks to get the mesh coordinates
    grid_x = tf.range(grid_size[1], dtype=tf.int32)
    grid_y = tf.range(grid_size[0], dtype=tf.int32)
    grid_x, grid_y = tf.meshgrid(grid_x, grid_y)
    x_offset = tf.reshape(grid_x, (-1, 1))
    y_offset = tf.reshape(grid_y, (-1, 1))
    x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
    # shape: [13, 13, 1, 2]
    x_y_offset = tf.cast(tf.reshape(x_y_offset, [grid_size[0], grid_size[1], 1, 2]), tf.float32)

    # get the absolute box coordinates on the feature_map
    box_centers = box_centers + x_y_offset
    # rescale to the original image scale
    box_centers = box_centers * ratio[::-1]

    # avoid getting possible nan value with tf.clip_by_value
    box_sizes = tf.clip_by_value(tf.exp(box_sizes), 1e-9, 50) * rescaled_anchors
    # rescale to the original image scale
    box_sizes = box_sizes * ratio[::-1]

    # shape: [N, 13, 13, 3, 4]
    # last dimension: (center_x, center_y, w, h)
    boxes = tf.concat([box_centers, box_sizes], axis=-1)

    # shape:
    # x_y_offset: [13, 13, 1, 2]
    # boxes: [N, 13, 13, 3, 4], rescaled to the original image scale
    # conf_logits: [N, 13, 13, 3, 1]
    # prob_logits: [N, 13, 13, 3, class_num]
    return x_y_offset, boxes, conf_logits, prob_logits

def reshape(result, class_num):
    x_y_offset, boxes, conf_logits, prob_logits = result
    grid_size = x_y_offset.shape.as_list()[:2]
    boxes = tf.reshape(boxes, [-1, grid_size[0] * grid_size[1] * 3, 4])
    conf_logits = tf.reshape(conf_logits, [-1, grid_size[0] * grid_size[1] * 3, 1])
    prob_logits = tf.reshape(prob_logits, [-1, grid_size[0] * grid_size[1] * 3, class_num])
    # shape: (take 416*416 input image and feature_map_1 for example)
    # boxes: [N, 13*13*3, 4]
    # conf_logits: [N, 13*13*3, 1]
    # prob_logits: [N, 13*13*3, class_num]
    return boxes, conf_logits, prob_logits

def predict(feature_map, anchors, img_size, class_num):
    '''
    Receive the returned feature_maps from `forward` function,
    the produce the output predictions at the test stage.
    '''
    reorg_results = reorg_layer(feature_map, anchors, img_size, class_num)

    boxes_list, confs_list, probs_list = [], [], []
    boxes, conf_logits, prob_logits = reshape(reorg_results, class_num)
    confs = tf.sigmoid(conf_logits)
    probs = tf.sigmoid(prob_logits)
    boxes_list.append(boxes)
    confs_list.append(confs)
    probs_list.append(probs)

    # collect results on three scales
    # take 416*416 input image for example:
    # shape: [N, (13*13+26*26+52*52)*3, 4]
    boxes = tf.concat(boxes_list, axis=1)
    # shape: [N, (13*13+26*26+52*52)*3, 1]
    confs = tf.concat(confs_list, axis=1)
    # shape: [N, (13*13+26*26+52*52)*3, class_num]
    probs = tf.concat(probs_list, axis=1)

    # center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
    # x_min = center_x - width / 2
    # y_min = center_y - height / 2
    # x_max = center_x + width / 2
    # y_max = center_y + height / 2

    # boxes = tf.concat([x_min, y_min, x_max, y_max], axis=-1)

    return boxes, confs, probs

anchors = np.array([[11, 22], [33, 44], [55, 66]])
class_num = 3
anchor_num = anchors.shape[0]
H = 5
W = 5
C = anchor_num * (5 + class_num)
img_h = H * 32
img_w = W * 32
feature_map = tf.reshape(tf.range(6.00, delta=0.01), [1, H, W, C])
boxes, confs, probs = predict(feature_map, anchors, np.array([img_h, img_w]), class_num)

np.set_printoptions(threshold=sys.maxsize)
with tf.Session() as sess:
    with open("anchors.txt", 'w') as f:
        f.write(str(anchors))
    with open("feature.txt", 'w') as f:
        f.write(str(feature_map.eval(session=sess)))
    with open("boxes.txt", 'w') as f:
        f.write(str(boxes.eval(session=sess)))
    with open("confs.txt", 'w') as f:
        f.write(str(confs.eval(session=sess)))
    with open("probs.txt", 'w') as f:
        f.write(str(probs.eval(session=sess)))
