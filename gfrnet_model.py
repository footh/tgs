import tensorflow as tf
from tgs import encoder
from tgs import model
from tgs import lovasz


class GatedFRefineUnet(model.BaseModel):
    """
        Unet model based on this paper: Gated Feedback Refinement Network for Coarse-to-Fine Dense Semantic Image
        Labeling
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        assert 'img_size' in config_dict['ext'].keys(), "img_size must be provided"
        assert 'encoder_l2_decay' in config_dict['ext'].keys(), "encoder_l2_decay must be provided"
        assert 'gate_channels' in config_dict['ext'].keys(), "gate_channels must be provided"

        self.name = 'gfrnet' if name is None else name

    def gate_unit(self, layer_i, layer_iplus1, channels=128, regularizer=None, training=True):
        """
            Calculate and return the gate unit, aka M-f in paper
        """
        i = self.conv2d_bn(layer_i, channels,
                           regularizer=regularizer, training=training)

        iplus1 = self.conv2d_bn(layer_iplus1, channels,
                                regularizer=regularizer, training=training)

        size = tf.shape(i)[1]
        iplus1 = tf.image.resize_bilinear(iplus1, (size, size), align_corners=True)

        Mf = tf.multiply(i, iplus1)

        return Mf

    def gated_refinement_unit(self, previous_gru, gate_unit, regularizer=None, training=True):

        size = tf.shape(gate_unit)[1]
        R = tf.image.resize_bilinear(previous_gru, (size, size), align_corners=True)

        m = self.conv2d_bn(gate_unit, 1, regularizer=regularizer, training=training, relu=False)

        gamma = tf.concat((R, m), axis=-1)

        PmRU = tf.layers.conv2d(gamma, 1, 3, padding='same', kernel_regularizer=regularizer)

        return PmRU

    def upsample(self, ds_layers, regularizer=None, training=True):
        """
            Decoder
        """
        assert(len(ds_layers) == 5)
        img_size = self.config_dict['ext']['img_size']
        gate_channels = self.config_dict['ext']['gate_channels']

        logits = []

        # Coarse prediction
        PmG = tf.layers.conv2d(ds_layers[4], 1, 3, padding='same', kernel_regularizer=regularizer)
        logits.append(tf.squeeze(PmG, axis=-1))

        M1 = self.gate_unit(ds_layers[3], ds_layers[4], gate_channels, regularizer=regularizer, training=training)
        PmRU1 = self.gated_refinement_unit(PmG, M1, regularizer=regularizer, training=training)
        logits.append(tf.squeeze(PmRU1, axis=-1))

        M2 = self.gate_unit(ds_layers[2], ds_layers[3], gate_channels // 2, regularizer=regularizer, training=training)
        PmRU2 = self.gated_refinement_unit(PmRU1, M2, regularizer=regularizer, training=training)
        logits.append(tf.squeeze(PmRU2, axis=-1))

        M3 = self.gate_unit(ds_layers[1], ds_layers[2], gate_channels // 4, regularizer=regularizer, training=training)
        PmRU3 = self.gated_refinement_unit(PmRU2, M3, regularizer=regularizer, training=training)
        logits.append(tf.squeeze(PmRU3, axis=-1))

        M4 = self.gate_unit(ds_layers[0], ds_layers[1], gate_channels // 8, regularizer=regularizer, training=training)
        PmRU4 = self.gated_refinement_unit(PmRU3, M4, regularizer=regularizer, training=training)

        # PmRU4 = tf.image.resize_bilinear(PmRU4, (img_size, img_size), align_corners=True)
        PmRU4 = tf.layers.conv2d_transpose(PmRU4, 1, 4, 2, padding='same', kernel_regularizer=regularizer)

        logits.append(tf.squeeze(PmRU4, axis=-1))

        return logits[::-1]

    def upsample2(self, ds_layers, regularizer=None, training=True):
        """
            Decoder
        """
        assert(len(ds_layers) == 4)
        img_size = self.config_dict['ext']['img_size']
        gate_channels = self.config_dict['ext']['gate_channels']

        logits = []

        # Coarse prediction
        PmG = tf.layers.conv2d(ds_layers[3], 1, 3, padding='same', kernel_regularizer=regularizer)
        logits.append(tf.squeeze(PmG, axis=-1))

        M1 = self.gate_unit(ds_layers[2], ds_layers[3], gate_channels, regularizer=regularizer, training=training)
        PmRU1 = self.gated_refinement_unit(PmG, M1, regularizer=regularizer, training=training)
        logits.append(tf.squeeze(PmRU1, axis=-1))

        M2 = self.gate_unit(ds_layers[1], ds_layers[2], gate_channels // 2, regularizer=regularizer, training=training)
        PmRU2 = self.gated_refinement_unit(PmRU1, M2, regularizer=regularizer, training=training)
        logits.append(tf.squeeze(PmRU2, axis=-1))

        M3 = self.gate_unit(ds_layers[0], ds_layers[1], gate_channels // 4, regularizer=regularizer, training=training)
        PmRU3 = self.gated_refinement_unit(PmRU2, M3, regularizer=regularizer, training=training)
        logits.append(tf.squeeze(PmRU3, axis=-1))

        PmRU3 = tf.image.resize_bilinear(PmRU4, (img_size, img_size), align_corners=True)
        logits.append(tf.squeeze(PmRU3, axis=-1))

        return logits[::-1]

    def build_model(self, inp, mode, regularizer=None):

        net = inp['img']

        training = (mode == tf.estimator.ModeKeys.TRAIN)

        with tf.variable_scope('encode'):
            ds_layers = encoder.build_resnet50_v1(net,
                                                  l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                                  is_training=training,
                                                  prefix=f'{self.name}/encode/')

        with tf.variable_scope('decode'):
            logits = self.upsample(ds_layers, regularizer=regularizer, training=training)

        return logits

    def logits_to_probs(self, logits):
        return tf.nn.sigmoid(logits[0])

    def loss_op(self, labels, logits):
        # w = [1.0, 1.0, 1.0, 1.0, 1.0]
        loss = tf.constant(0, tf.float32)
        for i, lg in enumerate(logits):
            tf.logging.info(lg.shape.as_list())
            s = tf.shape(lg)[1:3]
            lb = tf.expand_dims(labels, axis=-1)
            lb = tf.image.resize_bilinear(lb, s, align_corners=True)
            lb = tf.squeeze(lb, axis=-1)
            lb = tf.cast(tf.greater(lb, 0.5), tf.float32)

            bce_loss = tf.losses.sigmoid_cross_entropy(lb, lg)
            lov_loss = lovasz.lovasz_hinge(lg, lb)

            loss = loss + (bce_loss + lov_loss) / 2

        return loss
