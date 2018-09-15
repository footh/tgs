import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2
# from tensorflow.contrib.slim.python.slim.nets import inception_resnet_v2


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 include_root_block=True,
                 reuse=None,
                 scope='resnet_v1_50'):
    """
        ResNet-50 model of [1]. See resnet_v1() for arg and return description.
        (same as what's in slim library now but reversing the 1 stride to accommodate the unet model)
    """
    blocks = [
        resnet_v1.resnet_v1_block('block1', base_depth=64, num_units=3, stride=1),
        resnet_v1.resnet_v1_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v1.resnet_v1_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v1.resnet_v1_block('block4', base_depth=512, num_units=3, stride=2),
    ]

    return resnet_v1.resnet_v1(
        inputs,
        blocks,
        num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        include_root_block=include_root_block,
        reuse=reuse,
        scope=scope)


def build_resnet50_v1(img_input, l2_weight_decay=0.01, is_training=True, prefix=''):
    """
        Builds resnet50_v1 model from slim, with strides reversed.

        Returns the last five block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v1_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints[f'{prefix}resnet_v1_50/block3']
    block2 = endpoints[f'{prefix}resnet_v1_50/block2']
    block1 = endpoints[f'{prefix}resnet_v1_50/block1']
    conv1 = endpoints[f'{prefix}resnet_v1_50/conv1']

    return conv1, block1, block2, block3, block4


def build_resnet50_v1_custom(img_input, l2_weight_decay=0.01, is_training=True, prefix=''):
    """
        Builds resnet50_v1 model from slim, with strides reversed.

        Returns the last five block outputs to be used transposed convolution layers
    """
    regularizer = tf.contrib.layers.l2_regularizer(scale=l2_weight_decay)

    net = tf.layers.conv2d(img_input, 64, 3, padding='same', use_bias=False, kernel_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, training=is_training)
    net = tf.nn.relu(net)

    net = tf.layers.conv2d(net, 64, 3, padding='same', use_bias=False, kernel_regularizer=regularizer)
    net = tf.layers.batch_normalization(net, training=is_training)
    root = tf.nn.relu(net)

    net = tf.layers.max_pooling2d(root, 2, 2)

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v1_50(net, is_training=is_training, global_pool=False, include_root_block=False)

    block3 = endpoints[f'{prefix}resnet_v1_50/block3']
    block2 = endpoints[f'{prefix}resnet_v1_50/block2']
    block1 = endpoints[f'{prefix}resnet_v1_50/block1']

    return root, block1, block2, block3, block4


def resnet_v2_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
                 reuse=None,
                 scope='resnet_v2_50'):
    """
        ResNet-50 model of [1]. See resnet_v2() for arg and return description.
    """
    blocks = [
        resnet_v2.resnet_v2_block('block1', base_depth=64, num_units=3, stride=1),
        resnet_v2.resnet_v2_block('block2', base_depth=128, num_units=4, stride=2),
        resnet_v2.resnet_v2_block('block3', base_depth=256, num_units=6, stride=2),
        resnet_v2.resnet_v2_block('block4', base_depth=512, num_units=3, stride=2),
    ]

    return resnet_v2.resnet_v2(
        inputs,
        blocks,
        num_classes,
        is_training=is_training,
        global_pool=global_pool,
        output_stride=output_stride,
        include_root_block=True,
        reuse=reuse,
        scope=scope)


def build_resnet50_v2(img_input, l2_weight_decay=0.01, is_training=True, prefix=''):
    """
        Builds resnet50_v2 model from slim

        Returns the last five block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v2_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints[f'{prefix}resnet_v2_50/block3']
    block2 = endpoints[f'{prefix}resnet_v2_50/block2']
    block1 = endpoints[f'{prefix}resnet_v2_50/block1']
    conv1 = endpoints[f'{prefix}resnet_v2_50/conv1']

    return conv1, block1, block2, block3, block4


def build_inception_resnet_v2(img_input, l2_weight_decay=0.01, is_training=True, prefix=''):
    """
        Builds inception_resnet_v2 model from slim

        Returns the last five block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=l2_weight_decay)):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            block5, endpoints = inception_resnet_v2.inception_resnet_v2_base(img_input,
                                                                             align_feature_maps=True)

    # Conv2d_1a_3x3: [10, 64, 64, 32]
    # Conv2d_2a_3x3: [10, 64, 64, 32]
    # Conv2d_2b_3x3: [10, 64, 64, 64]
    # MaxPool_3a_3x3: [10, 32, 32, 64]
    # Conv2d_3b_1x1: [10, 32, 32, 80]
    # Conv2d_4a_3x3: [10, 32, 32, 192]
    # MaxPool_5a_3x3: [10, 16, 16, 192]
    # Mixed_5b: [10, 16, 16, 320]
    # Mixed_6a: [10, 8, 8, 1088]
    # PreAuxLogits: [10, 8, 8, 1088]
    # Mixed_7a: [10, 4, 4, 2080]
    # Conv2d_7b_1x1: [10, 4, 4, 1536]
    block4 = endpoints[f'{prefix}PreAuxLogits']
    block3 = endpoints[f'{prefix}Mixed_5b']
    block2 = endpoints[f'{prefix}Conv2d_4a_3x3']
    block1 = endpoints[f'{prefix}Conv2d_2b_3x3']

    return block1, block2, block3, block4, block5
