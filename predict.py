import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
from tgs import post_process as pp
import numpy as np
import os
import datetime

tf.logging.set_verbosity(tf.logging.INFO)

TRAIN_HOME_DIR = 'tgs/training-runs'
PREDICTION_DIR = 'tgs/predictions'
PREDICTION_RUN_DIR = 'run'
SUBMISSION_DIR = 'submission'
IDS_FILE = 'ids-test.npy'
# TODO: parameterize these
IMG_DIM = 101
SIZE = None


# Run-length encoding taken from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]

    run_lengths = []
    prev = -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))

        run_lengths[-1] += 1
        prev = b

    return run_lengths


def submission(ids, predictions, submission_file, prob_thresh=0.5, size=None):
    tf.logging.info('Writing submission file: %s' % submission_file)
    with tf.gfile.Open(submission_file, "w+") as f:
        f.write("id,rle_mask\n")

        for i, prediction in enumerate(predictions):
            tf.logging.info(f'Processing record {i + 1} with id: {ids[i]}')

            prediction = pp.threshold(prediction, prob_thresh=prob_thresh, size=size)
            rle = rle_encoding(prediction)

            line = f"{ids[i]}, {' '.join(map(str, rle))}\n"

            f.write(line)
            f.flush()


def build_resizes(cfg):
    min_padding = cfg.get('data.ext.min_padding')
    assert(min_padding is not None)
    resize_dim = cfg.get('data.ext.resize_dim')
    assert(resize_dim is not None)

    diff = resize_dim - IMG_DIM
    max_padding = diff - min_padding
    mid_padding = diff // 2

    # resizes = [
    #     [[min_padding, max_padding], [min_padding, max_padding], [0, 0]],
    #     [[max_padding, min_padding], [min_padding, max_padding], [0, 0]],
    #     [[min_padding, max_padding], [max_padding, min_padding], [0, 0]],
    #     [[max_padding, min_padding], [max_padding, min_padding], [0, 0]],
    #     [[mid_padding, diff - mid_padding], [mid_padding, diff - mid_padding], [0, 0]]
    # ]

    # resizes = [
    #     [[min_padding, max_padding], [mid_padding, diff - mid_padding], [0, 0]],
    #     [[max_padding, min_padding], [mid_padding, diff - mid_padding], [0, 0]]
    # ]

    # resizes = [
    #     [[mid_padding, diff - mid_padding], [min_padding, max_padding], [0, 0]],
    #     [[mid_padding, diff - mid_padding], [max_padding, min_padding], [0, 0]]
    # ]

    resizes = [
        [[mid_padding, diff - mid_padding], [mid_padding, diff - mid_padding], [0, 0]]
    ]

    return resizes


def record_count(file_exp):
    files = tf.gfile.Glob(file_exp)
    total = 0
    for f in files:
        total += sum([1 for _ in tf.python_io.tf_record_iterator(f)])

    return total


def reverse_augment(prediction, augment):
    if 'flip' in augment:
        if augment['flip']:
            prediction = np.fliplr(prediction)

    return prediction


def predict(cfg, checkpoint_path, augment=None, resize=None, hooks=None):

    tf.logging.info('Using data class: %s' % cfg.get('data.class'))
    dataset = data.DataInput.get(cfg.get('data.class'))(cfg.get('data'),
                                                        batch_size=cfg.get('batch_size'),
                                                        num_epochs=1,
                                                        label_cnt=cfg.get('model.label_cnt'))

    tf.logging.info('Using model: %s' % cfg.get('model.class'))
    model = m.BaseModel.get(cfg.get('model.class'))(cfg.get('model'))

    resize_method = cfg.get('data.ext.resize_method')
    params = {'l2_normalize': cfg.get('l2_normalize'),
              'resize_method': resize_method}
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=None, params=params)

    predictions = estimator.predict(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.PREDICT, augment, resize),
                                    predict_keys=['id', 'probabilities', resize_method],
                                    checkpoint_path=checkpoint_path,
                                    hooks=hooks)

    return predictions


def ensemble_unet(cfg, checkpoint_paths, augments):

    resize_method = cfg.get('data.ext.resize_method')
    resizes = [None]
    if resize_method == 'pad':
        resizes = build_resizes(cfg)

    total = record_count(cfg.get('data.test_file_pattern'))

    divisor = 0
    predictions = np.zeros((total, IMG_DIM, IMG_DIM), np.float32)
    ids = []
    for checkpoint_path in checkpoint_paths:
        tf.logging.info(f'Unet checkpoint path: {checkpoint_path}')
        for resize in resizes:
            tf.logging.info(f'Resize: {resize}')
            for augment in augments:
                tf.logging.info(f'Augment: {augment}')
                divisor += 1
                preds = predict(cfg, checkpoint_path, augment, resize)
                for i, pred in enumerate(preds):
                    tf.logging.info(f"Unet iteration {i}, id: {pred['id']}")
                    if divisor == 1:
                        ids.append(pred['id'].decode())

                    # NOTE: it is very important that this happens REVERSE the way the augment and resize happens in
                    # in data.py, ie. like popping a stack.
                    pred_ds = pp.downsample(pred['probabilities'], resize_method, pred[resize_method])
                    predictions[i] += reverse_augment(pred_ds, augment)

    predictions = predictions / divisor
    return ids, predictions


def ensemble_mask(cfg, checkpoint_paths, augments):

    total = record_count(cfg.get('data.test_file_pattern'))

    divisor = 0
    predictions = np.zeros(total, np.float32)
    ids = []
    for checkpoint_path in checkpoint_paths:
        tf.logging.info(f'Mask checkpoint path: {checkpoint_path}')
        for augment in augments:
            tf.logging.info(f'Augment: {augment}')
            divisor += 1
            preds = predict(cfg, checkpoint_path, augment)
            for i, pred in enumerate(preds):
                tf.logging.info(f"Mask iteration {i}, id: {pred['id']}")
                if divisor == 1:
                    ids.append(pred['id'].decode())
                predictions[i] += pred['probabilities'][0]

    predictions = predictions / divisor
    return ids, predictions


def run_prediction(save_file_name, checkpoint_paths, cfg,
                   checkpoint_paths_mask=None, cfg_mask=None, prob_thresh_mask=0.5,
                   save_ids=False):
    augments = [
        {'flip': 0},
        {'flip': 1}
    ]

    ids, predictions = ensemble_unet(cfg, checkpoint_paths, augments)

    if checkpoint_paths_mask is not None and cfg_mask is not None:
        ids_mask, predictions_mask = ensemble_mask(cfg_mask, checkpoint_paths_mask, augments)

        # Make sure the ids are the same!
        assert(np.array_equal(ids, ids_mask))

        predictions_mask_thresh = pp.threshold(predictions_mask, prob_thresh=prob_thresh_mask)
        predictions_mask_thresh = predictions_mask_thresh.reshape(predictions_mask_thresh.shape[0], 1, 1)

        predictions = predictions * ~predictions_mask_thresh

    save_file = os.path.join(PREDICTION_DIR, save_file_name)

    if save_ids:
        np.save(f'{save_file}-ids', ids)
    np.save(save_file, predictions)


def run_submission(save_file_name, prob_thresh=0.5):
    tf.logging.info(f'Using probability thresh: {prob_thresh}')

    run_dir = os.path.join(PREDICTION_DIR, PREDICTION_RUN_DIR)
    submission_dir = os.path.join(PREDICTION_DIR, SUBMISSION_DIR)

    ids = np.load(os.path.join(PREDICTION_DIR, IDS_FILE))

    pred_files = tf.gfile.Glob(os.path.join(run_dir, '*.npy'))
    assert(len(pred_files) > 0)

    predictions = np.load(pred_files[0])
    divisor = 1
    tf.logging.info(f'Added {pred_files[0]}')
    tf.logging.info(f'Divisor: {divisor}')
    if len(pred_files) > 1:
        for i in range(1, len(pred_files)):
            predictions += np.load(pred_files[i])
            divisor += 1
            tf.logging.info(f'Added {pred_files[i]}')
            tf.logging.info(f'Divisor: {divisor}')

    predictions = predictions / divisor

    save_file = os.path.join(submission_dir, save_file_name)

    submission(ids, predictions, f'{save_file}.csv', prob_thresh=prob_thresh, size=SIZE)


def main(_):
    save_file_name = f"{FLAGS.save_method}-{FLAGS.save_file_name}-" \
                     f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if FLAGS.save_method == 'prediction':
        tf.logging.info("Reading Unet config file...")
        cfg = config.Config(os.path.join(TRAIN_HOME_DIR, FLAGS.config_file))

        checkpoint_paths = [os.path.join(TRAIN_HOME_DIR, ckpt.strip()) for ckpt in FLAGS.checkpoint_paths.split(',')]
        assert (len(checkpoint_paths) > 0)

        checkpoint_paths_mask = None
        cfg_mask = None
        if FLAGS.checkpoint_paths_mask is not None and FLAGS.config_file_mask is not None:
            checkpoint_paths_mask = [ckpt.strip() for ckpt in FLAGS.checkpoint_paths_mask.split(',')]

            tf.logging.info("Reading Mask config file...")
            cfg_mask = config.Config(FLAGS.config_file_mask)

        run_prediction(save_file_name, checkpoint_paths, cfg, checkpoint_paths_mask, cfg_mask)
    else:
        run_submission(save_file_name, prob_thresh=FLAGS.prob_thresh)


tf.app.flags.DEFINE_string(
    'save_file_name', 'X',
    'The name of the save file (date will be automatically appended)')

tf.app.flags.DEFINE_string(
    'config_file', None,
    'File containing the unet configuration for this evaluation run')

tf.app.flags.DEFINE_string(
    'checkpoint_paths', None,
    'Comma delimted list of full paths to the checkpoints used to initialize the graph')

tf.app.flags.DEFINE_string(
    'config_file_mask', None,
    'File containing the mask configuration for this evaluation run')

tf.app.flags.DEFINE_string(
    'checkpoint_paths_mask', None,
    'Comma delimted list of full paths to the mask checkpoints used to initialize the graph')

tf.app.flags.DEFINE_string(
    'save_method', 'submission',
    "Method to save, either 'submission' or 'prediction' to save the submission file or raw predictions respectively")

tf.app.flags.DEFINE_float(
    'prob_thresh', 0.5,
    'Probability threshold to use when thresholding predictions')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

