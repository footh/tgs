import tensorflow as tf
from random import shuffle

VGG_RGB_MEANS = [123.68, 116.78, 103.94]


class DataInput(object):

    def __init__(self, config_dict, batch_size=1024, num_epochs=999999, label_cnt=1, preprocess=True, augment=True):

        assert config_dict is not None, "Config dictionary cannot be empty"
        # TODO: validate config entries?
        self.config_dict = config_dict

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.label_cnt = label_cnt
        self.preprocess = preprocess
        self.augment = augment

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
    def augment_image(img, mask):

        # Rotation
        rot = tf.random_uniform([], maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k=rot)
        mask = tf.image.rot90(mask, k=rot)

        # Flipping
        flip = tf.random_uniform([], maxval=2, dtype=tf.int32)
        img = tf.cond(tf.cast(flip, tf.bool), lambda: tf.image.flip_up_down(img), lambda: tf.identity(img))
        mask = tf.cond(tf.cast(flip, tf.bool), lambda: tf.image.flip_up_down(mask), lambda: tf.identity(mask))

        return img, mask

    def input_fn(self, mode):
        """
        Input function to be used in Estimator training
        """
        dataset = self.build_dataset(mode)

        # image_dim = self.config_dict['ext']['image_dim']
        resize_dim = self.config_dict['ext']['resize_dim']

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
            img = tf.image.resize_images(img, (resize_dim, resize_dim),
                                         method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

            if mode != tf.estimator.ModeKeys.PREDICT:
                mask = tf.image.decode_png(example['mask'])
                mask = tf.image.resize_images(mask, (resize_dim, resize_dim),
                                              method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            else:
                mask = tf.constant([])

            if self.augment:
                img, mask = self.augment_image(img, mask)

            # Tensorflow image operations don't work with 0 channel grayscales. So need to do this here.
            if mode != tf.estimator.ModeKeys.PREDICT:
                mask = tf.squeeze(mask, axis=-1)

            if self.preprocess:
                img = tf.subtract(tf.cast(img, tf.float32), VGG_RGB_MEANS)
                mask = tf.divide(tf.cast(mask, tf.float32), 255.)

            return example['id'], img, mask

        # Use `Dataset.map()` to build a pair of a feature dictionary and a label
        # tensor for each example.
        dataset = dataset.map(lambda rec: parser(rec, mode), num_parallel_calls=self.config_dict['parallel_calls'])
        if mode == tf.estimator.ModeKeys.TRAIN:
            dataset = dataset.shuffle(buffer_size=self.config_dict['shuf_buf'])
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.repeat(self.num_epochs)
        iterator = dataset.make_one_shot_iterator()

        img_id, img, mask = iterator.get_next()
        image_dict = {
            'id': img_id,
            'img': img
        }
        return image_dict, mask
