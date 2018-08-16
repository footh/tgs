import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
import os
import json

tf.logging.set_verbosity(tf.logging.INFO)

CONFIG_DIR = 'config'


class TrainingSessionRunHook(tf.train.SessionRunHook):

    def __init__(self, loss_scope=None):
        self.loss_scope = loss_scope

    def before_run(self, run_context):
        loss = tf.losses.get_losses(scope=self.loss_scope)[0]
        global_step = tf.train.get_global_step()
        return tf.train.SessionRunArgs([loss, global_step])

    def after_run(self, run_context, run_values):
        loss, global_step = run_values.results
        if (global_step + 1) % 2000 == 0:
            raise tf.errors.OutOfRangeError(None, None, "Test out of range")


def warm_start(cfg):
    settings = None
    if cfg.get('checkpoint.warm_start.checkpoint_path') is not None:
        warm_start_map = None
        if cfg.get('checkpoint.warm_start.var_map') is not None:
            with open(cfg.get('checkpoint.warm_start.var_map')) as f:
                warm_start_map = json.load(f)

        warm_start_var = None
        if cfg.get('checkpoint.warm_start.var_init') is not None:
            warm_start_var = cfg.get('checkpoint.warm_start.var_init')

        tf.logging.info('Warm starting with: %s' % cfg.get('checkpoint.warm_start_path'))
        settings = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=cfg.get('checkpoint.warm_start.checkpoint_path'),
                                                  vars_to_warm_start=warm_start_var,
                                                  var_name_to_prev_var_name=warm_start_map)

    return settings


def move_config(config_file, model_dir):
    """
        Move config file to model directory
    """
    tf.logging.info("Copying config file to model directory...")
    tf.gfile.MakeDirs(os.path.join(model_dir, CONFIG_DIR))
    new_config_file = os.path.join(model_dir, CONFIG_DIR, os.path.basename(config_file))
    tf.gfile.Copy(FLAGS.config_file, new_config_file, overwrite=True)

    return new_config_file


def train_and_eval(config_file, model_dir, hooks=None):

    tf.logging.info('Using Tensorflow version: %s' % tf.__version__)

    tf.logging.info("Reading config file...")
    c = config.Config(config_file)

    tf.logging.info('Using data class: %s' % c.get('data.class'))
    dataset = data.DataInput.get(c.get('data.class'))(c.get('data'),
                                                      batch_size=c.get('batch_size'),
                                                      label_cnt=c.get('model.label_cnt'))

    tf.logging.info('Using model: %s' % c.get('model.class'))
    model = m.BaseModel.get(c.get('model.class'))(c.get('model'))

    # Using time-based checkpoint saving. There is some weirdness with 'save_checkpoint_secs' and 'throttle_secs' below
    # that seems to have been resolved here:
    # https://github.com/tensorflow/tensorflow/commit/3edb609926f2521c726737fc1efeae1572dc6581#diff-bc4a1638bbcd88997adf5e723b8609c7
    # For now, setting throttle_secs to a minute less than save_checkpoint_secs which seems to work well.
    # TODO: would like to prevent checkpoint saving AFTER every eval
    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        save_checkpoints_steps=None,
                                        save_checkpoints_secs=c.get('checkpoint.save_seconds'),
                                        keep_checkpoint_max=c.get('checkpoint.keep'),
                                        log_step_count_steps=100 if c.get('log_steps') is None else c.get('log_steps'))

    params = {'learning_rate': c.get('learning_rate.base'),
              'learning_rate.exponential_decay': c.get('learning_rate.exponential_decay'),
              'learning_rate.cosine_decay': c.get('learning_rate.cosine_decay'),
              'l2_normalize': c.get('l2_normalize'),
              'l2_weight_decay': c.get('l2_weight_decay'),
              'adam_epsilon': c.get('optimizer.adam.epsilon'),
              'ema_decay': c.get('ema_decay'),
              'clip_grad_norm': c.get('clip_grad_norm'),
              'map_iou_thresholds': c.get('metric.map_iou.thresholds'),
              'map_iou_predthresh': c.get('metric.map_iou.pred_thresh')}

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       config=run_config,
                                       params=params,
                                       warm_start_from=warm_start(c))

    # tsrh = TrainingSessionRunHook(f"{model.name}/loss")
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.TRAIN),
                                        max_steps=c.get('train_steps'),
                                        hooks=hooks)

    # Per note above, need to use throttle_secs to affect timed evaluation. Checkpoint setting in are ignored.
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.EVAL),
                                      steps=c.get('valid_steps'), throttle_secs=c.get('checkpoint.save_seconds'))

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def main(_):
    tf.logging.info('Using Tensorflow version: %s' % tf.__version__)

    new_config_file = move_config(FLAGS.config_file, FLAGS.model_dir)

    train_and_eval(new_config_file, FLAGS.model_dir)


tf.app.flags.DEFINE_string(
    'model_dir', './yt8m/training-runs/test',
    'Directory to output the training checkpoints and events')

tf.app.flags.DEFINE_string(
    'config_file', './yt8m/config/exp-dense.yaml',
    'File containing the configuration for this training run')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

