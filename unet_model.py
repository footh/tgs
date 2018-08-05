import tensorflow as tf
from tgs import encoder
from tgs import model


class ResnetUnet(model.BaseModel):
    """
    Unet with Resnet decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        assert 'encoder_l2_decay' in config_dict['ext'].keys(), "encoder_l2_decay must be provided"
        assert 'process_channels' in config_dict['ext'].keys(), "process_channels must be provided"

        self.name = 'resnet_unet' if name is None else name

    def process_ds_layers(self, ds_layers, regularizer=None):
        """
            Process the downsample layers by running a convolution to get to desired output channels and upsampling each
            deeper layer to fuse with layer right above. Fused layers get a final 3x3 convolution.
        """
        ds_layers_out = []

        # NOTE: default activation for conv2d is tf.nn.relu
        # NOTE: default uniform xavier_initializer is used for the weights here.
        index = len(ds_layers) - 1
        while index >= 0:
            tf.logging.info(ds_layers[index].shape.as_list())
            net = tf.layers.conv2d(ds_layers[index], self.config_dict['ext']['process_channels'], 1,
                                   kernel_regularizer=regularizer, name=f'conv{index+1}')
            if len(ds_layers_out) > 0:
                up_layer = ds_layers_out[-1]
                up_size = [2 * l for l in up_layer.shape.as_list()[1:3]]
                up = tf.image.resize_images(up_layer, up_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                net = tf.add(net, up, name=f"fuse{index+1}")
                net = tf.layers.conv2d(net, self.config_dict['ext']['process_channels'], 3, padding='same',
                                       kernel_regularizer=regularizer, name=f'fuse_conv{index+1}')

            ds_layers_out.append(net)
            index -= 1

        return ds_layers_out[::-1]

    @staticmethod
    def residual_ds_layers(ds_layers, regularizer=None):
        """
            Perform a residual block on each incoming downsampled layer
        """
        residual_output = []

        for i, ds_layer in enumerate(ds_layers):
            net = tf.layers.conv2d(ds_layer, ds_layer.shape.as_list()[-1], 3, activation=tf.nn.relu, padding='same',
                                   kernel_regularizer=regularizer, name=f'conv_a{i+1}')

            net = tf.layers.conv2d(net, net.shape.as_list()[-1], 3, activation=tf.nn.relu, padding='same',
                                   kernel_regularizer=regularizer, name=f'conv_b{i+1}')

            net = tf.nn.relu(tf.add(ds_layer, net, name=f"fuse"))
            residual_output.append(net)

        return residual_output

    @staticmethod
    def upsample(ds_layers, img_size, regularizer=None):
        """
            Takes in a collection of downsampled layers, applies  transposed convolutions for each input layer returns
            the results.

            Returns the upsampled layers as an array

            kernel size calculated per here:
            http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
        """
        upsampled_outputs = []

        for i, ds_layer in enumerate(ds_layers):
            factor = img_size // ds_layer.shape.as_list()[1]
            kernel = 2 * factor - factor % 2

            tf.logging.debug(f"layer {i+1} kernel, stride (factor): {kernel, factor}")
            tf.logging.info(f"Layer shape: {ds_layer.shape.as_list()}")

            # Default uniform xavier_initializer is used for the weights here.
            net = tf.layers.conv2d_transpose(ds_layer, 1, kernel, factor, padding='same',
                                             kernel_regularizer=regularizer, name=f'tconv{i+1}')

            upsampled_outputs.append(net)

        return upsampled_outputs

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']
        img_size = net.shape.as_list()[1]
        if img_size != net.shape.as_list()[2]:
            raise ValueError('Image input must have equal dimensions')

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet50_v1(net,
                                                  l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                  is_training=training,
                                                  prefix=f'{self.name}/encode/')

        with tf.variable_scope('decode'):
            with tf.variable_scope('process_ds'):
                ds_layers = self.process_ds_layers(ds_layers, regularizer=regularizer)

            with tf.variable_scope('residual_ds'):
                ds_layers = self.residual_ds_layers(ds_layers, regularizer=regularizer)

            with tf.variable_scope('upsample'):
                us_layers = self.upsample(ds_layers, img_size, regularizer=regularizer)

                logits = tf.add_n(us_layers, name='fuse_us')

                logits = tf.squeeze(logits, axis=-1, name='squeeze')

        return logits

    def logits_to_probs(self, logits):
        return tf.nn.sigmoid(logits)

    def loss_op(self, labels, logits):
        return tf.losses.sigmoid_cross_entropy(labels, logits)
