import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.contrib.slim.python.slim.nets import resnet_v1
from tgs import model


class ResnetMask(model.BaseModel):
    """
        Model to predict whether image has a mask using Resnet encoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        assert 'encoder_l2_decay' in config_dict['ext'].keys(), "encoder_l2_decay must be provided"

        self.name = 'resnet_mask' if name is None else name

    def build_model(self, inp, mode, regularizer=None):
        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            with slim.arg_scope(resnet_v1.resnet_arg_scope(weight_decay=self.config_dict['ext']['encoder_l2_decay'])):
                net, _ = resnet_v1.resnet_v1_50(net,
                                                num_classes=None,
                                                is_training=training,
                                                global_pool=True)

        with tf.variable_scope('classify'):
            # net = tf.layers.max_pooling2d(net, net.shape.as_list()[1], 1)
            # net = tf.layers.conv2d(net, 1024, 1, kernel_regularizer=regularizer)
            net = tf.layers.conv2d(net, self.config_dict['label_cnt'], 1, kernel_regularizer=regularizer)
            logits = tf.squeeze(net, axis=(1, 2))

        return logits

    def logits_to_probs(self, logits):
        return tf.nn.sigmoid(logits)

    def loss_op(self, labels, logits):
        return tf.losses.sigmoid_cross_entropy(labels, logits)



