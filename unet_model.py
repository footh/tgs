import tensorflow as tf
from tgs import encoder
from tgs import model
from tgs import lovasz


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

    def process_ds_layers(self, ds_layers, regularizer=None, training=True):
        """
            Process the downsample layers by running a convolution to get to desired output channels and upsampling each
            deeper layer to fuse with layer right above. Fused layers get a final 3x3 convolution.
        """
        ds_layers_out = []

        with tf.variable_scope('process_ds'):
            index = len(ds_layers) - 1
            while index >= 0:
                net = self.conv2d_bn(ds_layers[index], self.config_dict['ext']['process_channels'], 1,
                                     padding='valid', regularizer=regularizer, training=training)

                if len(ds_layers_out) > 0:
                    up_layer = ds_layers_out[-1]
                    up_size = tf.shape(up_layer)[1:3] * 2
                    up = tf.image.resize_images(up_layer, up_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                    net = tf.add(net, up, name=f"fuse{index+1}")
                    net = self.conv2d_bn(net, self.config_dict['ext']['process_channels'], 3,
                                         regularizer=regularizer, training=training)

                ds_layers_out.append(net)
                index -= 1

        return ds_layers_out[::-1]

    def process_ds_layers_concat(self, ds_layers, regularizer=None, training=True):
        """
            Process the downsample layers by running a convolution to get to desired output channels and upsampling each
            deeper layer to fuse with layer right above. Fused layers get a final 3x3 convolution.
        """
        ds_layers_out = []

        with tf.variable_scope('process_ds'):
            index = len(ds_layers) - 1
            while index >= 0:
                net = self.conv2d_bn(ds_layers[index], self.config_dict['ext']['process_channels'], 1,
                                     padding='valid', regularizer=regularizer, training=training)

                if len(ds_layers_out) > 0:
                    up_layer = ds_layers_out[-1]
                    up_size = tf.shape(up_layer)[1:3] * 2
                    up = tf.image.resize_images(up_layer, up_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                    net = tf.concat([net, up], axis=-1, name=f"fuse_concat{index+1}")
                    net = self.conv2d_bn(net, self.config_dict['ext']['process_channels'], 3,
                                         regularizer=regularizer, training=training)

                ds_layers_out.append(net)
                index -= 1

        return ds_layers_out[::-1]

    def residual_ds_layers(self, ds_layers, regularizer=None, training=True):
        """
            Perform a residual block on each incoming downsampled layer
        """
        residual_output = []

        with tf.variable_scope('residual_ds'):
            for i, ds_layer in enumerate(ds_layers):
                net = self.conv2d_bn(ds_layer, self.config_dict['ext']['process_channels'], 3,
                                     regularizer=regularizer, training=training)

                net = self.conv2d_bn(net, self.config_dict['ext']['process_channels'], 3,
                                     relu=False, regularizer=regularizer, training=training)

                net = tf.nn.relu(tf.add(ds_layer, net, name=f"fuse"))
                residual_output.append(net)

        return residual_output

    def residual_ds_layers_exp(self, ds_layers, regularizer=None, training=True):
        """
            Perform a residual block on each incoming downsampled layer
        """

        residual_output = []
        with tf.variable_scope('residual_ds'):
            for i, ds_layer in enumerate(ds_layers):
                net = self.conv2d_bn(ds_layer, self.config_dict['ext']['process_channels'], 3,
                                     regularizer=regularizer, training=training)

                net = self.conv2d_bn(net, self.config_dict['ext']['process_channels'], 3,
                                     relu=False, regularizer=regularizer, training=training)

                net = tf.nn.relu(tf.add(ds_layer, net, name=f"fuse"))
                residual_output.append(net)

        residual_output_dilated = []
        with tf.variable_scope('residual_ds_dilated'):
            for i, ds_layer in enumerate(ds_layers):
                net = self.conv2d_bn(ds_layer, self.config_dict['ext']['process_channels'], 3, dilation=2,
                                     regularizer=regularizer, training=training)

                net = self.conv2d_bn(net, self.config_dict['ext']['process_channels'], 3, dilation=2,
                                     relu=False, regularizer=regularizer, training=training)

                net = tf.nn.relu(tf.add(ds_layer, net, name=f"dfuse"))
                residual_output_dilated.append(net)

        final_output = []
        with tf.variable_scope('residual_ds_final'):
            for i in range(len(ds_layers)):
                net = tf.concat([residual_output[i], residual_output_dilated[i]], axis=-1)
                net = self.conv2d_bn(net, self.config_dict['ext']['process_channels'], 3,
                                     regularizer=regularizer, training=training)

                final_output.append(net)

        return final_output

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
        # '6' comes from maximum downsamples (really it's 6 - 1 = 5)
        shapes = [img_size // (2 ** i) for i in range(6 - len(ds_layers), 6)]

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

    def decoder(self, ds_layers, regularizer=None, training=True):
        with tf.variable_scope('decode'):
            ds_layers = self.process_ds_layers(ds_layers, regularizer=regularizer, training=training)

            ds_layers = self.residual_ds_layers(ds_layers, regularizer=regularizer, training=training)

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
        weights = 1.
        if 'zero_mask_weight' in self.config_dict['ext'] and self.config_dict['ext']['zero_mask_weight'] is not None:
            tf.logging.info(f"Using zero_mask_weight: {self.config_dict['ext']['zero_mask_weight']}")
            zero_masks = tf.equal(tf.reduce_sum(labels, axis=(1, 2)), 0)
            nonzero_masks = tf.logical_not(zero_masks)

            weights = tf.cast(zero_masks, tf.float32) * self.config_dict['ext']['zero_mask_weight']
            weights = weights + tf.cast(nonzero_masks, tf.float32)
            weights = tf.expand_dims(weights, axis=-1)
            weights = tf.expand_dims(weights, axis=-1)
            labels_shape = tf.shape(labels)
            weights = tf.tile(weights, [1, labels_shape[1], labels_shape[2]])

        bce_loss = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)
        lov_loss = lovasz.lovasz_hinge(logits, labels)

        return (bce_loss + lov_loss) / 2


class ResnetV1Unet(UnetModel):
    """
        Unet with Resnet 50 v1 decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)

        self.name = 'resnet_unet' if name is None else name

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet50_v1(net,
                                                  l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                  is_training=training,
                                                  prefix=f'{self.name}/encode/')

        logits = self.decoder(ds_layers[1:], regularizer=regularizer, training=training)

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

        logits = self.decoder(ds_layers[1:], regularizer=regularizer, training=training)

        return logits


class InceptionResnetV2Unet(UnetModel):
    """
        Unet with Inception Resnet v2 decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)

        self.name = 'incepres_unet' if name is None else name

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_inception_resnet_v2(net,
                                                          l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                          is_training=training)

        logits = self.decoder(ds_layers, regularizer=regularizer, training=training)

        return logits


class Resnet34Unet(UnetModel):
    """
        Unet with Resnet 34 decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)

        self.name = 'resnet_34_unet' if name is None else name

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet34(net,
                                               l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                               is_training=training,
                                               prefix=f'{self.name}/encode/')

        logits = self.decoder(ds_layers[1:], regularizer=regularizer, training=training)

        return logits


class SimpleUnet(model.BaseModel):
    """
        Unet model that captures common functionality for upsampling
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        assert 'img_size' in config_dict['ext'].keys(), "img_size must be provided"
        assert 'encoder_l2_decay' in config_dict['ext'].keys(), "encoder_l2_decay must be provided"

        self.name = 'simple_unet' if name is None else name

    def upsample(self, ds_layers, regularizer=None, training=True):
        """
            Decoder
        """
        assert(len(ds_layers) == 5)

        root_size = self.config_dict['ext']['img_size'] // (2 ** 5)

        net = ds_layers[4]
        # root_sizex2048
        net = self.conv2d_bn(net, 1024, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 1024, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 2, root_size * 2), align_corners=True)
        # root_size*2x1024

        net = tf.concat((ds_layers[3], net), axis=-1)
        # root_size*2x2048
        net = self.conv2d_bn(net, 1024, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 512, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 4, root_size * 4), align_corners=True)
        # root_size*4x512

        net = tf.concat((ds_layers[2], net), axis=-1)
        # root_size*4x1024
        net = self.conv2d_bn(net, 512, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 256, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 8, root_size * 8), align_corners=True)
        # root_size*8x256

        net = tf.concat((ds_layers[1], net), axis=-1)
        # root_size*8x512
        net = self.conv2d_bn(net, 256, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 128, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 64, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 16, root_size * 16), align_corners=True)
        # root_size*16x64

        net = tf.concat((ds_layers[0], net), axis=-1)
        # root_size*16x128
        net = self.conv2d_bn(net, 64, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 32, regularizer=regularizer, training=training)
        # root_size*16x32

        # Since channels are remaining the same, could use bilinear initializer
        net = tf.layers.conv2d_transpose(net, 32, 4, 2, padding='same',
                                         kernel_regularizer=regularizer)
        net = tf.layers.batch_normalization(net, training=training)

        # root_size*32x32
        net = self.conv2d_bn(net, 32, kernel=1, regularizer=regularizer, training=training)
        # root_size*32x32
        logits = tf.layers.conv2d(net, 1, 1, kernel_regularizer=regularizer)

        # root_size*32x1

        return logits

    def upsample_light(self, ds_layers, regularizer=None, training=True):
        """
            Decoder
        """
        assert(len(ds_layers) == 5)

        root_size = self.config_dict['ext']['img_size'] // (2 ** 5)

        block4 = ds_layers[4]

        # root_sizex2048
        net = self.conv2d_bn(block4, 1024, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 512, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 2, root_size * 2), align_corners=True)
        # root_size*2x512

        block3 = self.conv2d_bn(ds_layers[3], 512, kernel=1, regularizer=regularizer, training=training)
        net = tf.concat((block3, net), axis=-1)
        # root_size*2x1024
        net = self.conv2d_bn(net, 512, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 256, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 4, root_size * 4), align_corners=True)
        # root_size*4x256

        block2 = self.conv2d_bn(ds_layers[2], 256, kernel=1, regularizer=regularizer, training=training)
        net = tf.concat((block2, net), axis=-1)
        # root_size*4x512
        net = self.conv2d_bn(net, 256, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 128, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 8, root_size * 8), align_corners=True)
        # root_size*8x128

        block1 = self.conv2d_bn(ds_layers[1], 128, kernel=1, regularizer=regularizer, training=training)
        net = tf.concat((block1, net), axis=-1)
        # root_size*8x256
        net = self.conv2d_bn(net, 128, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 64, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 16, root_size * 16), align_corners=True)
        # root_size*16x64

        net = tf.concat((ds_layers[0], net), axis=-1)
        # root_size*16x128
        net = self.conv2d_bn(net, 64, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, 32, regularizer=regularizer, training=training)
        # root_size*16x32

        # Since channels are remaining the same, could use bilinear initializer
        net = tf.layers.conv2d_transpose(net, 32, 4, 2, padding='same',
                                         kernel_regularizer=regularizer)
        net = tf.layers.batch_normalization(net, training=training)

        # root_size*32x32
        net = self.conv2d_bn(net, 32, kernel=1, regularizer=regularizer, training=training)
        # root_size*32x32
        logits = tf.layers.conv2d(net, 1, 1, kernel_regularizer=regularizer)
        # root_size*32x1

        return logits

    def upsample_lighter(self, ds_layers, regularizer=None, training=True):
        """
            Decoder
        """
        assert(len(ds_layers) == 5)

        root_size = self.config_dict['ext']['img_size'] // (2 ** 5)  # 8
        root_channels = 512

        block4 = self.conv2d_bn(ds_layers[4], root_channels, kernel=1, regularizer=regularizer, training=training)
        # root_sizex512
        net = self.conv2d_bn(block4, root_channels, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 2, regularizer=regularizer, training=training)
        net = tf.image.resize_nearest_neighbor(net, (root_size * 2, root_size * 2))
        # root_size*2x256

        block3 = self.conv2d_bn(ds_layers[3], root_channels // 2, kernel=1, regularizer=regularizer, training=training)
        net = tf.concat((block3, net), axis=-1)
        # root_size*2x512
        net = self.conv2d_bn(net, root_channels // 2, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 4, regularizer=regularizer, training=training)
        net = tf.image.resize_nearest_neighbor(net, (root_size * 4, root_size * 4))
        # root_size*4x128

        block2 = self.conv2d_bn(ds_layers[2], root_channels // 4, kernel=1, regularizer=regularizer, training=training)
        net = tf.concat((block2, net), axis=-1)
        # root_size*4x256
        net = self.conv2d_bn(net, root_channels // 4, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = tf.image.resize_nearest_neighbor(net, (root_size * 8, root_size * 8))
        # root_size*8x64

        block1 = self.conv2d_bn(ds_layers[1], root_channels // 8, kernel=1, regularizer=regularizer, training=training)
        net = tf.concat((block1, net), axis=-1)
        # root_size*8x128
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = tf.image.resize_nearest_neighbor(net, (root_size * 16, root_size * 16))
        # root_size*16x64

        top = ds_layers[0]
        net = tf.concat((top, net), axis=-1)
        # root_size*16x128
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        # root_size*16x64

        # Since channels are remaining the same, could use bilinear initializer
        net = tf.layers.conv2d_transpose(net, root_channels // 8, 4, 2, padding='same',
                                         kernel_regularizer=regularizer, use_bias=False)
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)

        # root_size*16x64
        net = self.conv2d_bn(net, root_channels // 8, kernel=1, regularizer=regularizer, training=training)
        # root_size*16x64
        logits = tf.layers.conv2d(net, 1, 1, kernel_regularizer=regularizer)
        # root_size*16x1

        return logits

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet50_v1(net,
                                                  l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                  is_training=training,
                                                  prefix=f'{self.name}/encode/')

        with tf.variable_scope('decode'):
            logits = self.upsample_lighter(ds_layers, regularizer=regularizer, training=training)

            logits = tf.squeeze(logits, axis=-1)

        return logits

    def logits_to_probs(self, logits):
        return tf.nn.sigmoid(logits)

    def loss_op(self, labels, logits):
        weights = 1.
        if 'zero_mask_weight' in self.config_dict['ext'] and self.config_dict['ext']['zero_mask_weight'] is not None:
            tf.logging.info(f"Using zero_mask_weight: {self.config_dict['ext']['zero_mask_weight']}")
            zero_masks = tf.equal(tf.reduce_sum(labels, axis=(1, 2)), 0)
            nonzero_masks = tf.logical_not(zero_masks)

            weights = tf.cast(zero_masks, tf.float32) * self.config_dict['ext']['zero_mask_weight']
            weights = weights + tf.cast(nonzero_masks, tf.float32)
            weights = tf.expand_dims(weights, axis=-1)
            weights = tf.expand_dims(weights, axis=-1)
            labels_shape = tf.shape(labels)
            weights = tf.tile(weights, [1, labels_shape[1], labels_shape[2]])

        bce_loss = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)
        lov_loss = lovasz.lovasz_hinge(logits, labels)

        return (bce_loss + lov_loss) / 2


class Simple34Unet(model.BaseModel):
    """
        Unet model that captures common functionality for upsampling
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        assert 'img_size' in config_dict['ext'].keys(), "img_size must be provided"
        assert 'encoder_l2_decay' in config_dict['ext'].keys(), "encoder_l2_decay must be provided"

        self.name = 'simple34_unet' if name is None else name

    def upsample(self, ds_layers, regularizer=None, training=True):
        """
            Decoder
        """
        assert(len(ds_layers) == 5)

        root_size = self.config_dict['ext']['img_size'] // (2 ** 4)

        # root_channels = tf.shape(ds_layers[4])[-1]
        root_channels = 512

        # root_sizex512
        net = self.conv2d_bn(ds_layers[4], root_channels, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 2, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 2, root_size * 2), align_corners=True)
        # root_size*2x256

        net = tf.concat((ds_layers[3], net), axis=-1)
        # root_size*2x512
        net = self.conv2d_bn(net, root_channels // 2, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 4, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 4, root_size * 4), align_corners=True)
        # root_size*4x128

        net = tf.concat((ds_layers[2], net), axis=-1)
        # root_size*4x256
        net = self.conv2d_bn(net, root_channels // 4, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = tf.image.resize_bilinear(net, (root_size * 8, root_size * 8), align_corners=True)
        # root_size*8x64

        net = tf.concat((ds_layers[1], net), axis=-1)
        # root_size*8x128
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = tf.image.resize_nearest_neighbor(net, (root_size * 16, root_size * 16))
        # root_size*16x64

        top = ds_layers[0]
        net = tf.concat((top, net), axis=-1)
        # root_size*16x128
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        net = self.conv2d_bn(net, root_channels // 8, regularizer=regularizer, training=training)
        # root_size*16x64

        # Since channels are remaining the same, could use bilinear initializer
        # net = tf.layers.conv2d_transpose(net, root_channels // 8, 4, 2, padding='same',
        #                                  kernel_regularizer=regularizer, use_bias=False)
        # net = tf.layers.batch_normalization(net, training=training)
        # net = tf.nn.relu(net)

        # root_size*32x64
        net = self.conv2d_bn(net, root_channels // 8, kernel=1, regularizer=regularizer, training=training)
        # root_size*32x64
        logits = tf.layers.conv2d(net, 1, 1, kernel_regularizer=regularizer)
        # root_size*32x1

        return logits

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet34(net,
                                               l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                               is_training=training,
                                               prefix=f'{self.name}/encode/')

        with tf.variable_scope('decode'):
            logits = self.upsample(ds_layers, regularizer=regularizer, training=training)

            logits = tf.squeeze(logits, axis=-1)

        return logits

    def logits_to_probs(self, logits):
        return tf.nn.sigmoid(logits)

    def loss_op(self, labels, logits):
        weights = 1.
        if 'zero_mask_weight' in self.config_dict['ext'] and self.config_dict['ext']['zero_mask_weight'] is not None:
            tf.logging.info(f"Using zero_mask_weight: {self.config_dict['ext']['zero_mask_weight']}")
            zero_masks = tf.equal(tf.reduce_sum(labels, axis=(1, 2)), 0)
            nonzero_masks = tf.logical_not(zero_masks)

            weights = tf.cast(zero_masks, tf.float32) * self.config_dict['ext']['zero_mask_weight']
            weights = weights + tf.cast(nonzero_masks, tf.float32)
            weights = tf.expand_dims(weights, axis=-1)
            weights = tf.expand_dims(weights, axis=-1)
            labels_shape = tf.shape(labels)
            weights = tf.tile(weights, [1, labels_shape[1], labels_shape[2]])

        bce_loss = tf.losses.sigmoid_cross_entropy(labels, logits, weights=weights)
        lov_loss = lovasz.lovasz_hinge(logits, labels)

        return (bce_loss + lov_loss) / 2
