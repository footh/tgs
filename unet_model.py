import tensorflow as tf
from tgs import encoder
from tgs import model


class ResNetUNet(model.BaseModel):
    """
    Unet with ResNet decoder
    """

    def __init__(self, config_dict, name=None):
        super().__init__(config_dict)
        # assert 'encoder_l2_decay' in config_dict['ext'].keys(), "Encoder l2 decay must be provided"

        self.name = 'renet_unet' if name is None else name

    def build_model(self, inp, mode, regularizer=None):
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        net = inp['img']

        ds_layers = encoder.build_resnet50_v1(net,
                                              l2_weight_decay=self.config_dict['ext']['encoder_l2_decay'],
                                              is_training=training)

    def logits_to_probs(self, logits):
        pass

    def loss_op(self, labels, logits):
        pass
