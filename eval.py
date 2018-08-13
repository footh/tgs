import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
from tgs import metric
import numpy as np
import os

tf.logging.set_verbosity(tf.logging.INFO)

ANALYZE_DIR = 'analyze'


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


def evaluate(config_file, checkpoint_path, hooks=None):
    tf.logging.info("Reading config file...")
    c = config.Config(config_file)

    tf.logging.info('Using data class: %s' % c.get('data.class'))
    dataset = data.DataInput.get(c.get('data.class'))(c.get('data'),
                                                      batch_size=c.get('batch_size'),
                                                      num_epochs=1,
                                                      label_cnt=c.get('model.label_cnt'),
                                                      augment=False)

    tf.logging.info('Using model: %s' % c.get('model.class'))
    model = m.BaseModel.get(c.get('model.class'))(c.get('model'))

    resize_method = c.get('data.ext.resize_method')
    params = {'l2_normalize': c.get('l2_normalize'),
              'map_iou_thresholds': c.get('metric.map_iou.thresholds'),
              'map_iou_predthresh': c.get('metric.map_iou.pred_thresh'),
              'resize_method': resize_method}
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=None, params=params)

    evaluation = estimator.evaluate(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.EVAL),
                                    steps=c.get('valid_steps'),
                                    hooks=hooks,
                                    checkpoint_path=checkpoint_path)

    return evaluation


def analyze(results, config_file, output_dir='.'):
    c = config.Config(config_file)
    resize_method = c.get('data.ext.resize_method')

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

    with tf.gfile.Open(os.path.join(output_dir, 'metrics.csv'), "w+") as f:
        f.write("id,map_iou,loss\n")
        for i in range(len(ids)):
            line = f"{ids[i].decode('utf-8')},{metrics[i][0]},{metrics[i][1]}\n"
            f.write(line)
            f.flush()


def main(_):
    if FLAGS.analyze:
        analyze_hook = AnalyzeEvaluationHook()
        evaluate(FLAGS.config_file, FLAGS.checkpoint_path, hooks=[analyze_hook])

        output_dir = os.path.join(os.path.dirname(FLAGS.checkpoint_path), ANALYZE_DIR)
        tf.gfile.MakeDirs(output_dir)
        analyze(analyze_hook.results_dict, FLAGS.config_file, output_dir=output_dir)
    else:
        evaluate(FLAGS.config_file, FLAGS.checkpoint_path)


tf.app.flags.DEFINE_string(
    'config_file', None,
    'File containing the configuration for this evaluation run')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Full path to the checkpoint used to initialize the graph')

tf.app.flags.DEFINE_boolean(
    'analyze', False,
    'Whether to run evaluation analysis')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

