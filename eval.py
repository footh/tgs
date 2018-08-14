import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
from tgs import analyze as a
import os

tf.logging.set_verbosity(tf.logging.INFO)


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


def main(_):
    if FLAGS.analyze:
        analyze_hook = a.AnalyzeEvaluationHook()
        evaluate(FLAGS.config_file, FLAGS.checkpoint_path, hooks=[analyze_hook])

        output_dir = os.path.dirname(FLAGS.checkpoint_path)
        a.analyze(analyze_hook.results_dict, FLAGS.config_file, output_dir=output_dir)
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
    'Whether to run the evaluation analysis')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

