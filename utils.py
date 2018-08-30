import tensorflow as tf
import numpy as np
import os
import json


def checkpoint_average(checkpoints, output_path):
    """
        Averages checkpoint weights provided in 'checkpoints' list and saves to 'output_path'
        TODO: will need to do it the way 'avg_checkpoint' does things. This way makes a HUGE .meta file
        presumably because constants are getting assigned to the variables. Using the 'placeholder' method, values are
        pumped in and hopefully not saved in the meta graph.
    """

    # Build variable value dict and initialize values to zero
    var_values = {}
    var_list = tf.train.list_variables(checkpoints[0])
    for name, shape in var_list:
        if not name.startswith('global_step') and not name.startswith('beta') and name.find('Adam') == -1:
            var_values[name] = np.zeros(shape, dtype=np.float32)

    # Reading tensors and calculating average
    for checkpoint in checkpoints:
        reader = tf.train.load_checkpoint(checkpoint)
        tf.logging.info(f'Read from checkpoint {checkpoint}')
        for name in var_values:
            tf.logging.info(f'Reading variable: {name}')
            tensor = reader.get_tensor(name)
            # print(f'Tensor sum: {np.sum(tensor)}')
            var_values[name] += (tensor / len(checkpoints))
            # print(f'var_values sum: {np.sum(var_values[name])}')

    # Building new variables and assign ops
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int64)
    assign_ops = []
    for name, value in var_values.items():
        var = tf.get_variable(name, shape=value.shape, dtype=tf.float32)
        assign_ops.append(tf.assign(var, tf.constant(value)))

    saver = tf.train.Saver(tf.global_variables())

    assign_all_op = tf.group(assign_ops)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(assign_all_op)
        tf.logging.info(f'Saving average checkpoint to: {output_path}')
        saver.save(sess, output_path, global_step=global_step)


def convert_ema(checkpoint_path):
    """
        Convert a checkpoint to the exponential moving average values
    """
    meta_path = f'{checkpoint_path }.meta'

    saver = tf.train.import_meta_graph(meta_path)
    sess = tf.InteractiveSession()
    saver.restore(sess, checkpoint_path)
    train_vars = tf.trainable_variables()

    cr = tf.train.NewCheckpointReader(checkpoint_path)
    ema_weights_op = tf.group([tf.assign(var, cr.get_tensor(f'{var.op.name}/ExponentialMovingAverage')) for var in train_vars])
    sess.run(ema_weights_op)
    
    path = os.path.dirname(checkpoint_path)
    fname = os.path.basename(checkpoint_path)
    saver.save(sess, os.path.join(path, f'ema-{fname}'))


def build_warm_start_map(checkpoint_path, prefix):
    """
        Build map of prefixed model var names to original names for use in warm start
        TODO: {'resnet_v1_50/mean_rgb', 'resnet_v1_50/logits/biases', 'resnet_v1_50/logits/weights'}
    """
    var_list = tf.train.list_variables(checkpoint_path)

    var_prefix_dict = {}
    for name, shape in var_list:
        # Removing global step variable
        # Removing training-specific moving statistics for batch norm
        # Removing any exponential moving average value
        # Removing any momentum values
        # Removing moving mean and variance
        if not name.startswith('global_step') \
           and name.find('BatchNorm/moving') == -1 \
           and name.find('ExponentialMovingAverage') == -1 \
           and name.find('Momentum') == -1 \
           and name.find('moving_mean') == -1 \
           and name.find('moving_variance') == -1:
            key = f'{prefix}{name}'
            var_prefix_dict[key] = name

    dir = os.path.dirname(checkpoint_path)
    fname = os.path.splitext(os.path.basename(checkpoint_path))[0]

    with open(os.path.join(dir, f'{fname}.json'), 'w') as f:
        json.dump(var_prefix_dict, f)

    return var_prefix_dict
