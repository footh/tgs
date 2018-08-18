import tensorflow as tf
from tgs import config
from tgs import metric
from tgs import model as m
import numpy as np
import os
import skimage
from PIL import Image

ANALYZE_DIR = 'analyze'
BOTTOM_K_DIR = 'bottom_k'
IMAGE_DIR = 'tgs/data/train/images'


def resize(arr, resize_method, resize_param):

    if resize_method == 'pad':
        top = resize_param[0, 0]
        bottom = resize_param[0, 1]
        left = resize_param[1, 0]
        right = resize_param[1, 1]
        r, c = arr.shape
        return arr[top:r - bottom, left:c - right]
    else:
        # TODO: interpolated resize
        return np.resize(arr, (resize_param, resize_param))


class AnalyzeEvaluationHook(tf.train.SessionRunHook):
    """
        Retrieves tensors useful for analyzing the evaluation run which are stored in the 'results_dict' property.
    """

    def __init__(self):
        results_dict = {'id': [], 'prediction': [], 'label': [], 'resize_param': []}
        self.results_dict = results_dict

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(tf.get_collection('analyze_eval'))

    def after_run(self, run_context, run_values):
        ids, preds, labels, resize_params = run_values.results

        self.results_dict['id'].extend(ids)
        self.results_dict['prediction'].extend(preds)
        self.results_dict['label'].extend(labels)
        self.results_dict['resize_param'].extend(resize_params)


def analyze(results, cfg, output_dir='.', bottom_k=10):
    """
        Given evaluation results performs analysis:
        1) Writes metrics csv
        2) Takes bottom k metric results and writes to image files
    """
    resize_method = cfg.get('data.ext.resize_method')

    ids = results['id']
    predictions = results['prediction']
    labels = results['label']
    resize_params = results['resize_param']

    # Resize to original dimensions
    for i in range(len(ids)):
        predictions[i] = resize(predictions[i], resize_method, resize_params[i])
        labels[i] = resize(labels[i], resize_method, resize_params[i])

    # Build map_iou and loss graph
    preds = tf.placeholder(tf.float32)
    lbls = tf.placeholder(tf.float32)
    preds_batch = tf.expand_dims(preds, axis=0)
    lbls_batch = tf.expand_dims(lbls, axis=0)
    map_iou = metric.map_iou(preds_batch, lbls_batch)
    loss = m.BaseModel.cross_entropy_loss(lbls_batch, preds_batch)

    metrics = []
    with tf.Session() as sess:
        for i in range(len(ids)):
            metrics.append(sess.run([map_iou, loss], feed_dict={preds: predictions[i], lbls: labels[i]}))

    output_dir = os.path.join(output_dir, ANALYZE_DIR)
    tf.gfile.MakeDirs(output_dir)

    with tf.gfile.Open(os.path.join(output_dir, 'metrics.csv'), "w+") as f:
        f.write("id,map_iou,loss\n")
        for i in range(len(ids)):
            line = f"{ids[i].decode('utf-8')},{metrics[i][0]},{metrics[i][1]}\n"
            f.write(line)
            f.flush()

    metrics = np.asarray(metrics)
    bottom_k_dir = os.path.join(output_dir, BOTTOM_K_DIR)
    tf.gfile.MakeDirs(bottom_k_dir)
    bottom_indices = np.argpartition(metrics[:, 0], bottom_k)[:bottom_k]

    for i in bottom_indices:
        img_id = ids[i].decode()
        pred_img = skimage.img_as_ubyte(predictions[i])
        pred_img = Image.fromarray(pred_img)
        pred_img.save(os.path.join(bottom_k_dir, f'{img_id}-pred.png'))

        mask_img = skimage.img_as_ubyte(labels[i])
        mask_img = Image.fromarray(mask_img)
        mask_img.save(os.path.join(bottom_k_dir, f'{img_id}-mask.png'))

        orig_img = Image.open(os.path.join(IMAGE_DIR, f'{img_id}.png'))
        orig_img.save(os.path.join(bottom_k_dir, f'{img_id}-orig.png'))
