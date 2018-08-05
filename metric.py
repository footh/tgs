import tensorflow as tf


def iou(predictions, labels, pred_thresh=0.5):
    """
        Calculates the IoU for each prediction in batch. The 'pred_thresh' argument determines whether a prediction
        probability is considered a positive value. Returns a bs x 1 tensor of IoU values

        In the case that the labels AND predictions have no positive values, the 'union' will be 0. The IoU will result
        in a 1.0 since the prediction accurately predicted there were no labels.
    """
    preds = tf.greater_equal(predictions, pred_thresh)
    labels_bool = tf.cast(labels, tf.bool)

    intersection = tf.reduce_sum(tf.cast(tf.logical_and(preds, labels_bool), tf.float32), axis=(1, 2))
    union = tf.reduce_sum(tf.cast(tf.logical_or(preds, labels_bool), tf.float32), axis=(1, 2))

    # If union is zero, prediction accurately predicted no labels, set intersection and union to 1.0
    union_zero = tf.equal(union, 0)
    ones = tf.ones(tf.shape(union), dtype=tf.float32)
    intersection = tf.where(union_zero, ones, intersection)
    union = tf.where(union_zero, ones, union)

    ious = intersection / union
    ious = tf.expand_dims(ious, axis=-1)
    tf.logging.debug(ious)
    return ious


def map_iou(predictions, labels,
            thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
            pred_thresh=0.5):
    """
        Mean average precision at iou thresholds. Calculates precision value at each IoU threshold value and averages
        both across thresholds and across the batch. Returns scalar result.

        Note: the 'precision' is essentially 1 or 0 (it passes the threshold or not) since there is only one mask per
        image for this contest
    """

    # Get the IoU for each item in batch and expand (tile) to length of thresholds
    ious = iou(predictions, labels, pred_thresh=pred_thresh)
    ious = tf.tile(ious, (1, len(thresholds)))

    # Expand (tile) the thresholds to the batch size
    bs = tf.shape(ious)[0]
    thresholds_bs = tf.tile(tf.expand_dims(thresholds, axis=0), (bs, 1))

    result = tf.reduce_mean(tf.cast(tf.greater_equal(ious, thresholds_bs), tf.float32))

    return result


def map_iou_metric(predictions, labels,
                   thresholds=(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95),
                   pred_thresh=0.5):
    """
    Returns a value operation and update operation for calculating the MAP at IoU thresholds in the style of a
    tensorflow metric.
    """
    with tf.variable_scope('metrics/map_iou'):
        value_op = map_iou(predictions, labels, thresholds=thresholds, pred_thresh=pred_thresh)
        return tf.metrics.mean(value_op)
