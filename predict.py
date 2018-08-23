import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
from tgs import post_process as pp
import numpy as np
import os
import datetime

tf.logging.set_verbosity(tf.logging.INFO)

SUBMISSION_DIR = 'submission'
IMG_DIM = 101


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


def predict(cfg, checkpoint_path, augment=None, hooks=None):

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

    predictions = estimator.predict(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.PREDICT, augment),
                                    predict_keys=['id', 'probabilities', resize_method],
                                    checkpoint_path=checkpoint_path,
                                    hooks=hooks)

    return predictions


def ensemble(cfg, checkpoint_paths, augments):

    resize_method = cfg.get('data.ext.resize_method')

    total = record_count(cfg.get('data.test_file_pattern'))

    divisor = 0
    predictions = np.zeros((total, IMG_DIM, IMG_DIM), np.float32)
    ids = []
    for checkpoint_path in checkpoint_paths:
        tf.logging.info(f'Checkpoint path: {checkpoint_path}')
        for augment in augments:
            tf.logging.info(f'Augment: {augment}')
            divisor += 1
            preds = predict(cfg, checkpoint_path, augment)
            for i, pred in enumerate(preds):
                tf.logging.info(f"Iteration {i}, id: {pred['id']}")
                if divisor == 1:
                    ids.append(pred['id'].decode())
                pred_rv = reverse_augment(pred['probabilities'], augment)
                predictions[i] += pp.downsample(pred_rv, resize_method, pred[resize_method])

    predictions = predictions / divisor

    return ids, predictions


def main(_):
    checkpoint_paths = [ckpt.strip() for ckpt in FLAGS.checkpoint_paths.split(',')]

    augments = [
        {'flip': 0},
        {'flip': 1}
    ]

    tf.logging.info("Reading config file...")
    cfg = config.Config(FLAGS.config_file)

    ids, predictions = ensemble(cfg, checkpoint_paths, augments)

    # Using the first checkpoint path as directory home to save files
    submission_dir = os.path.join(os.path.dirname(checkpoint_paths[0]), SUBMISSION_DIR)
    tf.gfile.MakeDirs(submission_dir)

    save_file_name = f"{FLAGS.save_method}-{FLAGS.save_file_name}-" \
                     f"{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    save_file_name = os.path.join(submission_dir, save_file_name)

    if FLAGS.save_method == 'prediction':
        np.save(f'{save_file_name}-ids', ids)
        np.save(save_file_name, predictions)
    else:
        submission(ids, predictions, f'{save_file_name}.csv', prob_thresh=0.5, size=20)


tf.app.flags.DEFINE_string(
    'save_file_name', 'X',
    'The name of the save file (date will be automatically appended)')

tf.app.flags.DEFINE_string(
    'config_file', None,
    'File containing the configuration for this evaluation run')

tf.app.flags.DEFINE_string(
    'checkpoint_paths', None,
    'Comma delimted list of full paths to the checkpoints used to initialize the graph')

tf.app.flags.DEFINE_string(
    'save_method', 'submission',
    'Method to save, either submission or prediction to save the submission file or raw predictions respectively')


FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

