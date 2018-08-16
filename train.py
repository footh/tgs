import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
import os
import json

tf.logging.set_verbosity(tf.logging.INFO)

CONFIG_DIR = 'config'


def warm_start(cfg):
    """
        Configures a warm start setting for use in warm starting a training session. Allows for a mapping of variables
        to warm start and a regex of variables to initialize.
    """
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


def build_dataset(cfg, epochs=999999):
    """
        Builds dataset for use in estimator training
    """
    tf.logging.info('Using data class: %s' % cfg.get('data.class'))
    dataset = data.DataInput.get(cfg.get('data.class'))(cfg.get('data'),
                                                        batch_size=cfg.get('batch_size'),
                                                        label_cnt=cfg.get('model.label_cnt'),
                                                        num_epochs=epochs)

    return dataset


def build_estimator(cfg, model_dir):
    """
        Builds estimator to be used in training
    """

    tf.logging.info('Using model: %s' % cfg.get('model.class'))
    model = m.BaseModel.get(cfg.get('model.class'))(cfg.get('model'))

    run_config = tf.estimator.RunConfig(model_dir=model_dir,
                                        save_checkpoints_steps=None,
                                        save_checkpoints_secs=cfg.get('checkpoint.save_seconds'),
                                        keep_checkpoint_max=cfg.get('checkpoint.keep'),
                                        log_step_count_steps=100 if cfg.get('log_steps') is None else cfg.get('log_steps'))

    params = {'learning_rate': cfg.get('learning_rate.base'),
              'learning_rate.exponential_decay': cfg.get('learning_rate.exponential_decay'),
              'learning_rate.cosine_decay': cfg.get('learning_rate.cosine_decay'),
              'l2_normalize': cfg.get('l2_normalize'),
              'l2_weight_decay': cfg.get('l2_weight_decay'),
              'adam_epsilon': cfg.get('optimizer.adam.epsilon'),
              'ema_decay': cfg.get('ema_decay'),
              'clip_grad_norm': cfg.get('clip_grad_norm'),
              'map_iou_thresholds': cfg.get('metric.map_iou.thresholds'),
              'map_iou_predthresh': cfg.get('metric.map_iou.pred_thresh')}

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       config=run_config,
                                       params=params,
                                       warm_start_from=warm_start(cfg))

    return estimator


def train_and_eval(cfg, dataset, estimator, eval_seconds=600, hooks=None):
    """
        Performs the estimator's train_and_evaluate method
    """

    train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.TRAIN),
                                        max_steps=cfg.get('train_steps'),
                                        hooks=hooks)

    # Evaluation starts at the minimum of an epoch end or the throttle_secs argument. So this must be coordinated with
    # the dataset num_epochs
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.EVAL),
                                      steps=cfg.get('valid_steps'), throttle_secs=eval_seconds)

    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def train(cfg, dataset, estimator, hooks=None):
    """
        Performs the estimators train method. Returns the estimator for chaining.
    """
    return estimator.train(dataset.input_fn, hook=hooks, steps=cfg.get('train_steps'))


def main(_):
    tf.logging.info('Using Tensorflow version: %s' % tf.__version__)

    new_config_file = move_config(FLAGS.config_file, FLAGS.model_dir)

    tf.logging.info("Reading config file...")
    cfg = config.Config(new_config_file)

    dataset = build_dataset(cfg)

    estimator = build_estimator(cfg, FLAGS.model_dir)

    train_and_eval(cfg, dataset, estimator)


tf.app.flags.DEFINE_string(
    'model_dir', './tgs/training-runs/test',
    'Directory to output the training checkpoints and events')

tf.app.flags.DEFINE_string(
    'config_file', './tgs/config/exp-resnetunet.yaml',
    'File containing the configuration for this training run')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

