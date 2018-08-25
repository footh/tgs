import tensorflow as tf
from random import shuffle
import math

VGG_RGB_MEANS = [123.68, 116.78, 103.94]


class DataInput(object):

    def __init__(self, config_dict, batch_size=1024, num_epochs=999999, label_cnt=1, preprocess=True):

        assert config_dict is not None, "Config dictionary cannot be empty"
        # TODO: validate config entries?
        self.config_dict = config_dict

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.label_cnt = label_cnt
        self.preprocess = preprocess

    @staticmethod
    def get(class_name):
        """
        Returns the model class object from the class name string passed in
        """
        # Though __name__ is 'tgs.data', the module returned is tgs. Calling 'data' returns this module
        module = __import__(__name__).data
        klass = getattr(module, class_name, None)
        return klass

    def build_dataset(self, mode):
        file_pattern = self.config_dict['train_file_pattern']
        if mode == tf.estimator.ModeKeys.EVAL:
            file_pattern = self.config_dict['valid_file_pattern']
        elif mode == tf.estimator.ModeKeys.PREDICT:
            file_pattern = self.config_dict['test_file_pattern']

        tf.logging.info('File pattern: %s' % file_pattern)

        filenames = tf.gfile.Glob(file_pattern)
        if mode == tf.estimator.ModeKeys.TRAIN:
            shuffle(filenames)
        dataset = tf.data.TFRecordDataset(filenames)

        return dataset

    def input_fn(self, mode):
        """
        Builds the model and returns the logits. Impmlemented by sub-classes.
        """
        raise NotImplementedError()


class ImageDataInput(DataInput):
    """
    Reads TFRecords image Examples.
    """
    @staticmethod
    def augment(img, mask, augment_dict=None):
        """
            Apply various random augmentations to image and mask
        """
        if augment_dict is None:
            return img, mask

        # Rotation
        if 'rotation' in augment_dict:
            if augment_dict['rotation'] is None:
                angle = tf.random_uniform([], minval=math.radians(-2), maxval=math.radians(2))
            else:
                angle = math.radians(augment_dict['rotation'])
            img = tf.contrib.image.rotate(img, angle)
            mask = tf.contrib.image.rotate(mask, angle)

        if 'rot90' in augment_dict:
            if augment_dict['rot90'] is None:
                rot = tf.random_uniform([], maxval=4, dtype=tf.int32)
            else:
                rot = augment_dict['rot90']
            img = tf.image.rot90(img, k=rot)
            mask = tf.image.rot90(mask, k=rot)

        # Shearing
        if 'shear' in augment_dict:
            if augment_dict['shear'] is None:
                sx = tf.divide(tf.cast(tf.random_uniform([], minval=90, maxval=101, dtype=tf.int32), tf.float32), tf.constant(100.))
                sy = tf.divide(tf.cast(tf.random_uniform([], minval=90, maxval=101, dtype=tf.int32), tf.float32), tf.constant(100.))
            else:
                sx, sy = augment_dict['shear']

            s_vec = tf.stack([sx, 1. - sx, 0., 1. - sy, sy, 0., 0., 0.])
            s_vec = tf.expand_dims(s_vec, axis=0)
            img = tf.contrib.image.transform(img, s_vec)
            mask = tf.contrib.image.transform(mask, s_vec)

        # Flipping
        if 'flip' in augment_dict:
            flip = augment_dict['flip']
            if flip is None:
                flip = tf.random_uniform([], maxval=2, dtype=tf.int32)
            img = tf.cond(tf.cast(flip, tf.bool), lambda: tf.image.flip_left_right(img), lambda: tf.identity(img))
            mask = tf.cond(tf.cast(flip, tf.bool), lambda: tf.image.flip_left_right(mask), lambda: tf.identity(mask))

        return img, mask

    def resize(self, img, resize_param=None, min_padding=5):
        """
            Resize an image using various methods
        """
        resize_dim = self.config_dict['ext']['resize_dim']
        orig_dim = tf.shape(img)[0]

        if self.config_dict['ext']['resize_method'] == 'pad':
            diff = resize_dim - orig_dim
            pad_var = diff - (min_padding * 2)

            if resize_param is not None:
                paddings = resize_param
            else:
                top = tf.random_uniform([], maxval=pad_var, dtype=tf.int32) + min_padding
                bottom = diff - top

                left = tf.random_uniform([], maxval=pad_var, dtype=tf.int32) + min_padding
                right = diff - left

                paddings = tf.reshape(tf.stack([top, bottom, left, right, 0, 0]), (3, 2))

            img = tf.pad(img, paddings, "SYMMETRIC")
            param = paddings
        elif self.config_dict['ext']['resize_method'] == 'pad-fixed':

            if resize_param is not None:
                paddings = resize_param
            else:
                top = 13
                bottom = 14

                left = 13
                right = 14

                paddings = tf.reshape(tf.stack([top, bottom, left, right, 0, 0]), (3, 2))

            img = tf.pad(img, paddings, "SYMMETRIC")
            param = paddings
        else:
            img = tf.image.resize_images(img, (resize_dim, resize_dim),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            param = orig_dim

        return img, param

    def input_fn(self, mode, augment_dict=None):
        """
        Input function to be used in Estimator training
        (ignore_augment is used to ignore the augment dict for augmenting data. Useful for train_and_evaluate where
        (the augmentation can be turned off on evaluation via the input_fn)
        """
        dataset = self.build_dataset(mode)

        # Use `tf.parse_single_example()` to extract data from a `tf.Example`
        # protocol buffer, and perform any additional per-record preprocessing.
        def parser(record, mode):

            # Build feature map to parse example
            feature_map = {
                'id': tf.FixedLenFeature([], tf.string),
                'img': tf.FixedLenFeature([], tf.string)
            }

            if mode != tf.estimator.ModeKeys.PREDICT:
                feature_map['mask'] = tf.FixedLenFeature([], tf.string)

            example = tf.parse_single_example(record, feature_map)

            img = tf.image.decode_png(example['img'])
            img, resize_param = self.resize(img)

            if mode != tf.estimator.ModeKeys.PREDICT:
                mask = tf.image.decode_png(example['mask'])
                mask, _ = self.resize(mask, resize_param)
            else:
                mask = tf.constant([[[0]]])

            if augment_dict is not None:
                img, mask = self.augment(img, mask, augment_dict)

            # Tensorflow image operations don't work with 0 channel grayscales. So need to do this here.
            if mode != tf.estimator.ModeKeys.PREDICT:
                mask = tf.squeeze(mask, axis=-1)

            if self.preprocess:
                img = tf.subtract(tf.cast(img, tf.float32), VGG_RGB_MEANS)
                mask = tf.divide(tf.cast(mask, tf.float32), 255.)

            return example['id'], img, mask, resize_param

        # Use `Dataset.map()` to build a pair of a feature dictionary and a label
        # tensor for each example.
        dataset = dataset.map(lambda rec: parser(rec, mode), num_parallel_calls=self.config_dict['parallel_calls'])
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=self.config_dict['shuf_buf'])
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        iterator = dataset.make_one_shot_iterator()

        img_id, img, mask, resize_param = iterator.get_next()
        image_dict = {
            'id': img_id,
            'img': img,
            self.config_dict['ext']['resize_method']: resize_param
        }
        return image_dict, mask
