import tensorflow as tf
import numpy as np
import os
from random import shuffle


def _byteslist_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _raw_image(f):
    with open(f, 'rb') as file:
        img = file.read()
    return img


def convert_to_sparse(labels):
    values = np.where(labels)[0]
    n = len(values)

    dense_shape = np.asarray([1, n])

    x = np.zeros(n, dtype=np.int64)
    y = np.asarray(range(n))
    indices = np.asarray(list(zip(x, y)))

    return tf.SparseTensorValue(indices, values, dense_shape)


def build_example(img_id, img, mask):
    feature = {
        'id': _byteslist_feature([img_id]),
        'img': _byteslist_feature([img]),
        'mask': _byteslist_feature([mask])
    }
    features = tf.train.Features(feature=feature)
    return tf.train.Example(features=features)


def write_examples(examples, output_file):
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    with tf.python_io.TFRecordWriter(output_file, options=opts) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def to_tfrecord(input_pattern, shards=40, output_dir='tfrecord', shuf=False):

    files = tf.gfile.Glob(input_pattern)
    if shuf:
        shuffle(files)
    base_dir = os.path.join(os.path.dirname(files[0]), '..')
    mask_dir = os.path.join(base_dir, 'masks')
    out_dir = os.path.join(base_dir, output_dir)

    tf.gfile.MakeDirs(out_dir)

    file_shards = np.array_split(files, shards)

    cnt = 0
    for i, file_shard in enumerate(file_shards):
        examples = []
        for f in file_shard:
            cnt += 1
            tf.logging.info(f'Iteration {cnt}, processing file: {f}')

            img_id, ext = os.path.splitext(os.path.basename(f))
            img = _raw_image(f)
            mask = _raw_image(os.path.join(mask_dir, f'{img_id}{ext}'))

            examples.append(build_example(img_id.encode(), img, mask))

        write_examples(examples, os.path.join(out_dir, f'tgs{i:02}'))


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
    global_step = tf.Variable(0, name="global_step", trainable=False, dtype=tf.int32)
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
