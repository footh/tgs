import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
from tgs import analyze as a
import os

tf.logging.set_verbosity(tf.logging.INFO)


def evaluate(cfg, checkpoint_path, hooks=None):

    tf.logging.info('Using data class: %s' % cfg.get('data.class'))
    dataset = data.DataInput.get(cfg.get('data.class'))(cfg.get('data'),
                                                        batch_size=cfg.get('batch_size'),
                                                        num_epochs=1,
                                                        label_cnt=cfg.get('model.label_cnt'))

    tf.logging.info('Using model: %s' % cfg.get('model.class'))
    model = m.BaseModel.get(cfg.get('model.class'))(cfg.get('model'))

    resize_method = cfg.get('data.ext.resize_method')
    params = {'l2_normalize': cfg.get('l2_normalize'),
              'map_iou_thresholds': cfg.get('metric.map_iou.thresholds'),
              'map_iou_predthresh': cfg.get('metric.map_iou.pred_thresh'),
              'resize_method': resize_method}
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=None, params=params)

    evaluation = estimator.evaluate(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.EVAL),
                                    steps=cfg.get('valid_steps'),
                                    hooks=hooks,
                                    checkpoint_path=checkpoint_path)

    return evaluation


def main(_):
    tf.logging.info("Reading config file...")
    cfg = config.Config(FLAGS.config_file)

    if FLAGS.analyze:
        analyze_hook = a.AnalyzeEvaluationHook()
        evaluate(cfg, FLAGS.checkpoint_path, hooks=[analyze_hook])

        output_dir = os.path.dirname(FLAGS.checkpoint_path)
        a.analyze(analyze_hook.results_dict, cfg, output_dir=output_dir)
    else:
        evaluate(cfg, FLAGS.checkpoint_path)


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

