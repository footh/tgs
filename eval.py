import tensorflow as tf
from tgs import config
from tgs import data
from tgs import model as m
import numpy as np
import os
import datetime
import skimage
from PIL import Image

tf.logging.set_verbosity(tf.logging.INFO)

SUBMISSION_DIR = 'submission'


def submission(predictions, submission_file, top_k=20):
    tf.logging.info('Writing submission file: %s' % submission_file)
    with tf.gfile.Open(submission_file, "w+") as f:
        f.write("id,rle_mask\n")

        for i, prediction in enumerate(predictions):
            tf.logging.info('Processing record %s with id: %s' % (i + 1, prediction['ids']))
            top_indices = np.argpartition(prediction['probabilities'], -top_k)[-top_k:]
            preds = prediction['probabilities'][top_indices]
            preds = list(zip(top_indices, preds))
            preds = sorted(preds, key=lambda p: -p[1])
            line = prediction['ids'].decode('utf-8') + ',' + ' '.join(
                "%i %g" % (label, score) for (label, score) in preds) + "\n"

            f.write(line)
            f.flush()


def main(_):
    tf.logging.info("Reading config file...")
    c = config.Config(FLAGS.config_file)

    tf.logging.info('Using data class: %s' % c.get('data.class'))
    dataset = data.DataInput.get(c.get('data.class'))(c.get('data'),
                                                      batch_size=c.get('batch_size'),
                                                      label_cnt=c.get('model.label_cnt'),
                                                      num_epochs=1,
                                                      augment=False)

    tf.logging.info('Using model: %s' % c.get('model.class'))
    model = m.BaseModel.get(c.get('model.class'))(c.get('model'))

    params = {'l2_normalize': c.get('l2_normalize')}
    estimator = tf.estimator.Estimator(model_fn=model.model_fn, config=None, params=params)

    predictions = estimator.predict(input_fn=lambda: dataset.input_fn(tf.estimator.ModeKeys.PREDICT),
                                    predict_keys=['id', 'probabilities'],
                                    checkpoint_path=FLAGS.checkpoint_path)

    submission_dir = os.path.join(os.path.dirname(FLAGS.checkpoint_path), SUBMISSION_DIR)
    tf.gfile.MakeDirs(submission_dir)
    submission_file_name = 'submission-%s-%s.csv' % \
                           (FLAGS.submission_file, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    submission_file_name = os.path.join(submission_dir, submission_file_name)
    # submission(predictions, submission_file_name, top_k=c.get('metric.gap.top_k'))

    for i, prediction in enumerate(predictions):
        if i > 10:
            break
        img_id = prediction['id'].decode()
        tf.logging.info(f"prediction id: {img_id}")

        mask_pred = prediction['probabilities']
        mask_pred = skimage.img_as_ubyte(mask_pred > 0.2)
        mask_gt = Image.open(f'tgs/data/train/masks/{img_id}.png')
        mask_gt = np.asarray(mask_gt.resize((128, 128))).astype(np.uint8)
        tf.logging.info(np.max(mask_pred.dtype))
        tf.logging.info(np.max(mask_gt.dtype))
        combo = np.concatenate((mask_pred, mask_gt), axis=-1)

        tf.gfile.MakeDirs('/tmp/tgs')
        Image.fromarray(combo).save(f"/tmp/tgs/{img_id}-p.png")


tf.app.flags.DEFINE_string(
    'submission_file', None,
    'The name of the submission file to save (date will be automatically appended)')

tf.app.flags.DEFINE_string(
    'config_file', None,
    'File containing the configuration for this evaluation run')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'Full path to the checkpoint used to initialize the graph')

FLAGS = tf.app.flags.FLAGS

if __name__ == '__main__':
    tf.app.run()

