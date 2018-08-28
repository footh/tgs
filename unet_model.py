import tensorflow as tf
from tgs import encoder
from tgs import model


class UnetModel(model.BaseModel):
    """
        Unet model that captures common functionality for upsampling
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        assert 'img_size' in config_dict['ext'].keys(), "img_size must be provided"
        assert 'encoder_l2_decay' in config_dict['ext'].keys(), "encoder_l2_decay must be provided"
        assert 'process_channels' in config_dict['ext'].keys(), "process_channels must be provided"

        self.name = 'unet' if name is None else name

    def process_ds_layers(self, ds_layers, regularizer=None):
        """
            Process the downsample layers by running a convolution to get to desired output channels and upsampling each
            deeper layer to fuse with layer right above. Fused layers get a final 3x3 convolution.
        """
        ds_layers_out = []

        with tf.variable_scope('process_ds'):
            # NOTE: default activation for conv2d is tf.nn.relu
            # NOTE: default uniform xavier_initializer is used for the weights here.
            index = len(ds_layers) - 1
            while index >= 0:
                net = tf.layers.conv2d(ds_layers[index], self.config_dict['ext']['process_channels'], 1,
                                       kernel_regularizer=regularizer, name=f'conv{index+1}')
                if len(ds_layers_out) > 0:
                    up_layer = ds_layers_out[-1]
                    up_size = tf.shape(up_layer)[1:3] * 2
                    up = tf.image.resize_images(up_layer, up_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                    net = tf.add(net, up, name=f"fuse{index+1}")
                    net = tf.layers.conv2d(net, self.config_dict['ext']['process_channels'], 3, padding='same',
                                           kernel_regularizer=regularizer, name=f'fuse_conv{index+1}')

                ds_layers_out.append(net)
                index -= 1

        return ds_layers_out[::-1]

    def residual_ds_layers(self, ds_layers, regularizer=None):
        """
            Perform a residual block on each incoming downsampled layer
        """
        residual_output = []

        with tf.variable_scope('residual_ds'):
            for i, ds_layer in enumerate(ds_layers):
                net = tf.layers.conv2d(ds_layer, self.config_dict['ext']['process_channels'], 3,
                                       activation=tf.nn.relu, padding='same',
                                       kernel_regularizer=regularizer, name=f'conv_a{i+1}')

                net = tf.layers.conv2d(net, self.config_dict['ext']['process_channels'], 3,
                                       activation=tf.nn.relu, padding='same',
                                       kernel_regularizer=regularizer, name=f'conv_b{i+1}')

                net = tf.nn.relu(tf.add(ds_layer, net, name=f"fuse"))
                residual_output.append(net)

        return residual_output

    def upsample(self, ds_layers, regularizer=None):
        """
            Takes in a collection of downsampled layers, applies  transposed convolutions for each input layer returns
            the results.

            Returns the upsampled layers as an array

            kernel size calculated per here:
            http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
        """
        upsampled_outputs = []

        img_size = self.config_dict['ext']['img_size']
        shapes = [img_size // (2 ** i) for i in range(2, 6)]

        with tf.variable_scope('upsample'):
            for i, ds_layer in enumerate(ds_layers):
                factor = img_size // shapes[i]
                kernel = 2 * factor - factor % 2

                # tf.logging.debug(f"layer {i+1} kernel, stride (factor): {kernel, factor}")
                # tf.logging.info(f"Layer shape: {ds_layer.shape.as_list()}")

                # Default uniform xavier_initializer is used for the weights here.
                net = tf.layers.conv2d_transpose(ds_layer, 1, kernel, factor, padding='same',
                                                 kernel_regularizer=regularizer, name=f'tconv{i+1}')

                upsampled_outputs.append(net)

        return upsampled_outputs

    def decoder(self, ds_layers, regularizer=None):
        with tf.variable_scope('decode'):
            ds_layers = self.process_ds_layers(ds_layers, regularizer=regularizer)

            ds_layers = self.residual_ds_layers(ds_layers, regularizer=regularizer)

            us_layers = self.upsample(ds_layers, regularizer=regularizer)

            with tf.variable_scope('upsample'):
                logits = tf.add_n(us_layers, name='fuse_us')

                logits = tf.squeeze(logits, axis=-1, name='squeeze')

        return logits

    def build_model(self, inp, mode, regularizer=None):
        raise NotImplementedError()

    def logits_to_probs(self, logits):
        return tf.nn.sigmoid(logits)

    def loss_op(self, labels, logits):
        return tf.losses.sigmoid_cross_entropy(labels, logits)


class ResnetV1Unet(UnetModel):
    """
        Unet with Resnet 50 v1 decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)

        self.name = 'resnet_unet' if name is None else name

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        img_size = self.config_dict['ext']['img_size']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet50_v1(net,
                                                  l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                  is_training=training,
                                                  prefix=f'{self.name}/encode/')

        logits = self.decoder(ds_layers, regularizer=regularizer)

        return logits


class ResnetV2Unet(UnetModel):
    """
        Unet with Resnet 50 v2 decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)

        self.name = 'resnetv2_unet' if name is None else name

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet50_v2(net,
                                                  l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                  is_training=training,
                                                  prefix=f'{self.name}/encode/')

        logits = self.decoder(ds_layers, regularizer=regularizer)

        return logits
