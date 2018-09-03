from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tensorflow.contrib.slim.python.slim.nets import resnet_v2


def resnet_v1_50(inputs,
                 num_classes=None,
                 is_training=True,
                 global_pool=True,
                 output_stride=None,
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
        include_root_block=True,
        reuse=reuse,
        scope=scope)


def build_resnet50_v1(img_input, l2_weight_decay=0.01, is_training=True, prefix=''):
    """
        Builds resnet50_v1 model from slim, with strides reversed.

        Returns the last four block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v1_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints[f'{prefix}resnet_v1_50/block3']
    block2 = endpoints[f'{prefix}resnet_v1_50/block2']
    block1 = endpoints[f'{prefix}resnet_v1_50/block1']
    conv1 = endpoints[f'{prefix}resnet_v1_50/conv1']

    return conv1, block1, block2, block3, block4


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

        Returns the last three block outputs to be used transposed convolution layers
    """

    with slim.arg_scope(resnet_v2.resnet_arg_scope(weight_decay=l2_weight_decay)):
        block4, endpoints = resnet_v2_50(img_input, is_training=is_training, global_pool=False)

    block3 = endpoints[f'{prefix}resnet_v2_50/block3']
    block2 = endpoints[f'{prefix}resnet_v2_50/block2']
    block1 = endpoints[f'{prefix}resnet_v2_50/block1']
    conv1 = endpoints[f'{prefix}resnet_v2_50/conv1']

    return conv1, block1, block2, block3, block4
