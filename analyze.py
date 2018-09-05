import tensorflow as tf
from tgs import metric
from tgs import model as m
from tgs import post_process
import numpy as np
import os
import skimage
from PIL import Image

ANALYZE_DIR = 'analyze'
BOTTOM_K_DIR = 'bottom_k'
IMAGE_DIR = 'tgs/data/raw/images'
MASK_DIR = 'tgs/data/raw/masks'


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


def analyze_unet(results, cfg, output_dir='.', bottom_k=20):
    """
        Given Unet evaluation results performs analysis:
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
        predictions[i] = post_process.downsample(predictions[i], resize_method, resize_params[i])
        labels[i] = post_process.downsample(labels[i], resize_method, resize_params[i])

    # Build map_iou and loss graph
    preds = tf.placeholder(tf.float32)
    lbls = tf.placeholder(tf.float32)
    preds_batch = tf.expand_dims(preds, axis=0)
    lbls_batch = tf.expand_dims(lbls, axis=0)
    map_iou = metric.map_iou(preds_batch, lbls_batch, pred_thresh=None)
    loss = m.BaseModel.cross_entropy_loss(lbls_batch, preds_batch)

    metrics = []
    thresholds = np.asarray(range(2, 20)) / 20.
    with tf.Session() as sess:
        for i in range(len(ids)):
            metrics_id = [sess.run(loss, feed_dict={preds: predictions[i], lbls: labels[i]})]
            for thresh in thresholds:
                pred = np.asarray(post_process.threshold(predictions[i], prob_thresh=thresh), np.float32)
                metrics_id.append(sess.run(map_iou, feed_dict={preds: pred, lbls: labels[i]}))

            metrics.append(metrics_id)

    metrics = np.asarray(metrics)

    output_dir = os.path.join(output_dir, ANALYZE_DIR)
    tf.gfile.MakeDirs(output_dir)

    metric_means = np.mean(metrics, axis=0)
    with tf.gfile.Open(os.path.join(output_dir, 'metrics.csv'), "w+") as f:
        map_iou_header = ','.join([f'map_iou{t}' for t in thresholds])
        f.write(f'id,loss,{map_iou_header}\n')
        for i in range(len(ids)):
            line = f"{ids[i].decode('utf-8')},{','.join(map(str, metrics[i]))}\n"
            f.write(line)
            f.flush()
        f.write(f"averages,{','.join(map(str, metric_means))}")

    bottom_k_dir = os.path.join(output_dir, BOTTOM_K_DIR)
    tf.gfile.MakeDirs(bottom_k_dir)
    # Ordering by worst mean ious
    bottom_indices = np.argpartition(np.mean(metrics[:, 1:], axis=-1), bottom_k)[:bottom_k]

    for i in bottom_indices:
        img_id = ids[i].decode()
        pred_img = skimage.img_as_ubyte(predictions[i])
        pred_img = Image.fromarray(pred_img)
        pred_img.save(os.path.join(bottom_k_dir, f'{img_id}-pred.png'))

        pred5_img = skimage.img_as_ubyte(predictions[i] > 0.5)
        pred5_img = Image.fromarray(pred5_img)
        pred5_img.save(os.path.join(bottom_k_dir, f'{img_id}-pred5.png'))

        mask_img = skimage.img_as_ubyte(labels[i])
        mask_img = Image.fromarray(mask_img)
        mask_img.save(os.path.join(bottom_k_dir, f'{img_id}-mask.png'))

        orig_img = Image.open(os.path.join(IMAGE_DIR, f'{img_id}.png'))
        orig_img.save(os.path.join(bottom_k_dir, f'{img_id}-orig.png'))

        omask_img = Image.open(os.path.join(MASK_DIR, f'{img_id}.png'))
        omask_img.save(os.path.join(bottom_k_dir, f'{img_id}-omask.png'))


def analyze_mask(results, cfg, output_dir='.', bottom_k=10):
    """
        Given mask evaluation results performs analysis:
        1) Writes metrics csv
        2) Takes bottom k metric results and writes to image files
        TODO: between this and unet, common code can be refactored out
    """
    ids = results['id']
    predictions = results['prediction']
    labels = results['label']

    # Build map_iou and loss graph
    preds = tf.placeholder(tf.float32)
    lbls = tf.placeholder(tf.float32)
    preds_batch = tf.expand_dims(preds, axis=0)
    lbls_batch = tf.expand_dims(lbls, axis=0)
    accuracy = metric.accuracy(preds_batch, lbls_batch, pred_thresh=None)
    loss = m.BaseModel.cross_entropy_loss(lbls_batch, preds_batch)

    metrics = []
    thresholds = np.asarray(range(2, 20)) / 20.
    with tf.Session() as sess:
        for i in range(len(ids)):
            metrics_id = [sess.run(loss, feed_dict={preds: predictions[i], lbls: labels[i]})]
            for thresh in thresholds:
                pred = np.asarray(post_process.threshold(predictions[i], prob_thresh=thresh), np.float32)
                metrics_id.append(sess.run(accuracy, feed_dict={preds: pred, lbls: labels[i]}))

            metrics.append(metrics_id)

    metrics = np.asarray(metrics)

    output_dir = os.path.join(output_dir, ANALYZE_DIR)
    tf.gfile.MakeDirs(output_dir)

    metric_means = np.mean(metrics, axis=0)
    with tf.gfile.Open(os.path.join(output_dir, 'metrics.csv'), "w+") as f:
        acc_header = ','.join([f'acc{t}' for t in thresholds])
        f.write(f'id,loss,{acc_header}\n')
        for i in range(len(ids)):
            line = f"{ids[i].decode('utf-8')},{','.join(map(str, metrics[i]))}\n"
            f.write(line)
            f.flush()
        f.write(f"averages,{','.join(map(str, metric_means))}")

    bottom_k_dir = os.path.join(output_dir, BOTTOM_K_DIR)
    tf.gfile.MakeDirs(bottom_k_dir)
    # Ordering by worst accuracy
    bottom_indices = np.argpartition(np.mean(metrics[:, 1:], axis=-1), bottom_k)[:bottom_k]

    for i in bottom_indices:
        img_id = ids[i].decode()
        prob = predictions[i][0]
        lbl = labels[i][0]

        orig_img = Image.open(os.path.join(IMAGE_DIR, f'{img_id}.png'))
        orig_img.save(os.path.join(bottom_k_dir, f'{img_id}-img-{prob:.4f}.png'))

        omask_img = Image.open(os.path.join(MASK_DIR, f'{img_id}.png'))
        omask_img.save(os.path.join(bottom_k_dir, f'{img_id}-mask-{lbl:.4f}.png'))
