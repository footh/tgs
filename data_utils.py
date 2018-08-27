import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import model_selection as ms
import os
import shutil
from random import shuffle

RAW_DIR = 'raw'


def _remove_files(src):
    """
        Remove files from src directory (train, test, etc) and sub-directories
    """
    if os.path.isfile(src):
        os.unlink(src)
    elif os.path.isdir(src):
        # map lazy evaluates so must wrap in list to force evaluation
        list(map(_remove_files, [os.path.join(src, fi) for fi in os.listdir(src)]))


def _copy_train_file(img_id, src, base_dir='tgs/data'):
    """
        Copy training files based on 'src' to appropriate directory
    """
    img_dir = os.path.join(base_dir, src, 'images')
    mask_dir = os.path.join(base_dir, src, 'masks')
    tf.gfile.MakeDirs(img_dir)
    tf.gfile.MakeDirs(mask_dir)

    source = os.path.join(base_dir, RAW_DIR, 'images', f'{img_id}.png')
    dest = os.path.join(img_dir, f'{img_id}.png')
    shutil.copy2(source, dest)

    source = os.path.join(base_dir, RAW_DIR, 'masks', f'{img_id}.png')
    dest = os.path.join(mask_dir, f'{img_id}.png')
    shutil.copy2(source, dest)


def _copy_train_files(img_ids, src, base_dir='tgs/data'):
    for img_id in img_ids:
        _copy_train_file(img_id, src, base_dir=base_dir)


def _byteslist_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _int64list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _floatlist_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _raw_image(f):
    with open(f, 'rb') as file:
        img = file.read()
    return img


def convert_to_sparse(labels):
    values = np.where(labels)[0]
    n = len(values)

    dense_shape = np.asarray([1, n])

    x = np.zeros(n, dtype=np.int64)
    y = np.asarray(range(n))
    indices = np.asarray(list(zip(x, y)))

    return tf.SparseTensorValue(indices, values, dense_shape)


def build_example(img_id, img, mask):
    feature = {
        'id': _byteslist_feature([img_id]),
        'img': _byteslist_feature([img]),
    }

    if mask is not None:
        feature['mask'] = _byteslist_feature([mask])

    features = tf.train.Features(feature=feature)
    return tf.train.Example(features=features)


def write_examples(examples, output_file):
    opts = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
    with tf.python_io.TFRecordWriter(output_file, options=opts) as writer:
        for example in examples:
            writer.write(example.SerializeToString())


def to_tfrecord(input_pattern, shards=40, output_dir='tfrecord', shuf=False, with_mask=True):

    files = tf.gfile.Glob(input_pattern)
    if shuf:
        shuffle(files)
    base_dir = os.path.join(os.path.dirname(files[0]), '..')
    mask_dir = os.path.join(base_dir, 'masks')
    out_dir = os.path.join(base_dir, output_dir)

    tf.gfile.MakeDirs(out_dir)

    file_shards = np.array_split(files, shards)

    cnt = 0
    for i, file_shard in enumerate(file_shards):
        examples = []
        for f in file_shard:
            cnt += 1
            tf.logging.info(f'Iteration {cnt}, processing file: {f}')

            img_id, ext = os.path.splitext(os.path.basename(f))
            img = _raw_image(f)
            mask = None
            if with_mask:
                mask = _raw_image(os.path.join(mask_dir, f'{img_id}{ext}'))

            examples.append(build_example(img_id.encode(), img, mask))

        write_examples(examples, os.path.join(out_dir, f'tgs{i:02}'))


def train_stats(base_dir='tgs/data', img_dim=101, bins=10, generate=True):
    """
        Return the train stats dataframe, default params will generate a new one
    """

    if generate:
        train_df = pd.read_csv(f'{base_dir}/train.csv', index_col='id', usecols=[0])

        # images = [np.array(Image.open(f'{base_dir}/{RAW_DIR}/images/{idx}.png')) for idx in train_df.index]
        masks = [np.array(Image.open(f'{base_dir}/{RAW_DIR}/masks/{idx}.png')) / 65535. for idx in train_df.index]

        train_df["coverage"] = list(map(lambda img: np.sum(img) / pow(img_dim, 2), masks))

        # Split coverage percentage into bins (bins count is 'bins + 1' as it includes that 0 bin)
        def cov_to_bin(val, b=bins):
            for i in range(0, b + 1):
                if val <= i / b:
                    return i

        train_df["coverage_bin"] = train_df.coverage.map(cov_to_bin)

        train_df.to_csv(f'{base_dir}/train_stats.csv')
    else:
        train_df = pd.read_csv(f'{base_dir}/train_stats.csv')

    return train_df


def build_sets(test_size, base_dir='tgs/data', train_stats_df=None, gen_stats=True, seed=1, out_pf=''):
    """
        Build training and validation sets
    """

    if train_stats_df is None:
        train_stats_df = train_stats(base_dir=base_dir, generate=gen_stats)

    id_train, id_val = ms.train_test_split(train_stats_df.id,
                                           test_size=test_size, stratify=train_stats_df.coverage_bin, random_state=seed)

    _remove_files(os.path.join(base_dir, f'train{out_pf}'))
    _remove_files(os.path.join(base_dir, f'valid{out_pf}'))
    for img_id in id_train:
        _copy_train_file(img_id, f'train{out_pf}', base_dir=base_dir)

    for img_id in id_val:
        _copy_train_file(img_id, f'valid{out_pf}', base_dir=base_dir)


def kfolds(splits, base_dir='tgs/data', train_stats_df=None, gen_stats=False, seed=1):
    """
        Split the data into stratified folds and copies them to the appropriate directories
    """
    if train_stats_df is None:
        train_stats_df = train_stats(base_dir=base_dir, generate=gen_stats)

    train_dir = os.path.join(base_dir, 'train-f')
    valid_dir = os.path.join(base_dir, 'valid-f')
    tf.logging.info(f'Removing previous folds...')
    if os.path.exists(train_dir) and os.path.isdir(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(valid_dir) and os.path.isdir(valid_dir):
        shutil.rmtree(valid_dir)

    ids = np.asarray(train_stats_df.id)
    classes = np.asarray(train_stats_df.coverage_bin)

    skf = ms.StratifiedKFold(splits, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(skf.split(ids, classes)):

        train_ids = ids[train_idx]
        train_dir = os.path.join('train-f', str(fold + 1))
        _copy_train_files(train_ids, train_dir, base_dir=base_dir)
        to_tfrecord(os.path.join(base_dir, train_dir, 'images', '*.png'), shards=len(train_ids) // 100)

        val_ids = ids[val_idx]
        val_dir = os.path.join('valid-f', str(fold + 1))
        _copy_train_files(val_ids, val_dir, base_dir=base_dir)
        to_tfrecord(os.path.join(base_dir, val_dir, 'images', '*.png'), shards=len(val_ids) // 100)
