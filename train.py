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
            with tf.gfile.Open(cfg.get('checkpoint.warm_start.var_map')) as f:
                warm_start_map = json.load(f)

        warm_start_var = '.*'
        if cfg.get('checkpoint.warm_start.var_init') is not None:
            warm_start_var = cfg.get('checkpoint.warm_start.var_init')

        tf.logging.info('Warm starting with: %s' % cfg.get('checkpoint.warm_start.checkpoint_path'))
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
                                        save_checkpoints_steps=cfg.get('checkpoint.save_steps'),
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
              'reduce_grad': cfg.get('reduce_grad')}

    if cfg.get('metric.accuracy') is not None:
        params['accuracy'] = cfg.get('metric.accuracy')

    if cfg.get('metric.map_iou') is not None:
        params['map_iou'] = cfg.get('metric.map_iou')

    estimator = tf.estimator.Estimator(model_fn=model.model_fn,
                                       config=run_config,
                                       params=params,
                                       warm_start_from=warm_start(cfg))

    return estimator


def train_and_eval(cfg, dataset, estimator, hooks=None):
    """
        Performs the estimator's train_and_evaluate method
    """
    # augment = {'rotation': None, 'shear': None, 'flip': None, 'rot90': None}
    # augment = {'flip': None, 'rot90': None}
    augment = {'flip': None}
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.TRAIN, augment),
                                        max_steps=cfg.get('train_steps'),
                                        hooks=hooks)

    # Evaluation always seems to occur after first checkpoint save which is controlled by save_checkpoint_* arguments
    # in estimator RunConfig. Then, throttle_secs kicks in where it will wait a minimum of this many seconds before an
    # evaluation is run again (after a RunConfig configured checkpoint save). Setting to 1 second means the RunConfig
    # save_checkpoint_* arguments will control evaluation triggers.
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.EVAL),
                                      steps=cfg.get('valid_steps'), throttle_secs=1)

    return tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def train(cfg, dataset, estimator, hooks=None):

    while True:
        # Will the estimator's RunConfig save_checkpoint_* settings cause training to stop or do I need to set this
        # 'steps' parameter. Either way, I think this should be set to the save_checkpoint_secs for connsistency. But
        # if I want to vary checkpoint save steps (ie. as loss
        estimator.train(lambda: dataset.input_fn(tf.estimator.ModeKeys.TRAIN),
                        steps=cfg.get('train_steps'),
                        hook=hooks)

        # Use these to reset learning rate and prevent warm start on future calls and set save checkpoint steps
        estimator._params = None
        estimator._warm_start_settings = None
        estimator._config.save_checkpoints_steps = None

        evaluation = estimator.evaluate(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.EVAL),
                                        steps=cfg.get('valid_steps'))


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

tf.app.flags.DEFINE_boolean(
    'train_and_eval', False,
    'Switch to run tensorflow train_and_evaluate')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

