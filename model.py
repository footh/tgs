import tensorflow as tf
from tgs import metric
import numpy as np


class BaseModel(object):
    """
    Inherit from this class when implementing new models.
    """

    def __init__(self, config_dict):
        assert config_dict is not None, "Config dictionary cannot be empty"
        # TODO: validate config entries?
        self.config_dict = config_dict
        self.name = 'base'

    @staticmethod
    def get(class_name):
        """
        Returns the model class object from the class name string passed in
        """
        from tgs import unet_model
        modules = [unet_model]
        classes = [getattr(module, class_name, None) for module in modules]
        klass = next(cls for cls in classes if cls)
        return klass

    @staticmethod
    def trainable_param_count():
        """
        Returns count of trainable parameters
        """
        return np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

    @staticmethod
    def cross_entropy_loss(labels, predictions):
        """
        Cross entropy loss for when predictions are already squashed to [0,1]
        """
        epsilon = 10e-6
        float_labels = tf.cast(labels, tf.float32)
        cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + \
                             (1 - float_labels) * tf.log(1 - predictions + epsilon)
        cross_entropy_loss = tf.negative(cross_entropy_loss)
        return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

    @staticmethod
    def clip_gradient_norms(grads_and_vars, max_norm):
        clipped_grads_and_vars = []
        for grad, var in grads_and_vars:
            if grad is not None:
                if isinstance(grad, tf.IndexedSlices):
                    tmp = tf.clip_by_norm(grad.values, max_norm)
                    grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
                else:
                    grad = tf.clip_by_norm(grad, max_norm)
            clipped_grads_and_vars.append((grad, var))

        return clipped_grads_and_vars

    def debug_graph(self, batch_size=512, feature_size=1152):
        """
        Useful debugging information for graphs
        TODO: print shapes
        """
        inp = tf.placeholder(tf.float32, [batch_size, feature_size])
        logits = self.build_model(inp)
        return self.trainable_param_count(), logits

    def build_model(self, inp, mode, regularizer=None):
        """
        Builds the model and returns the logits. Impmlemented by sub-classes.
        """
        raise NotImplementedError()

    def logits_to_probs(self, logits):
        """
        Converts logits to probabilities. Implemented by sub-classes.
        """
        raise NotImplementedError()

    def loss_op(self, labels, logits):
        """
        Calculates and returns the loss operation. Implemented by sub-classes
        """
        raise NotImplementedError()

    def model_fn(self, features, labels, mode, params):
        """
        Function used as input to an Estimator. Returns an EstimatorSpec configured based on mode
        """
        if 'l2_normalize' in params and params['l2_normalize']:
            features['img'] = tf.nn.l2_normalize(features['img'], axis=-1)
        tf.summary.histogram('model/inp', features['img'])

        regularizer = None
        if 'l2_weight_decay' in params and params['l2_weight_decay'] is not None:
            regularizer = tf.contrib.layers.l2_regularizer(scale=params['l2_weight_decay'])

        # Build graph and retrieve the logits tensor
        with tf.variable_scope(self.name):
            logits = self.build_model(features, mode, regularizer)

        predicted_probs = self.logits_to_probs(logits)

        predictions = {
            'id': features['id'],
            'probabilities': predicted_probs
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            spec = tf.estimator.EstimatorSpec(mode, predictions=predictions)
        else:
            with tf.variable_scope(f'{self.name}/loss'):
                loss = self.loss_op(labels, logits)

            # Add regularization
            loss = loss + tf.losses.get_regularization_loss()

            if mode == tf.estimator.ModeKeys.EVAL:
                map_iou_metric = metric.map_iou_metric(predicted_probs, labels,
                                                       thresholds=params['map_iou_thresholds'],
                                                       pred_thresh=params['map_iou_predthresh'])
                metrics = {'map_iou': map_iou_metric}

                spec = tf.estimator.EstimatorSpec(mode,
                                                  predictions=predictions,
                                                  loss=loss,
                                                  eval_metric_ops=metrics,
                                                  evaluation_hooks=None)
            else:
                assert mode == tf.estimator.ModeKeys.TRAIN

                global_step = tf.train.get_global_step()

                lr = params['learning_rate']
                if params['learning_rate.cosine_decay'] is not None:
                    tf.logging.info('learning_rate: cosine_decay_restarts')
                    learning_rate = tf.train.cosine_decay_restarts(lr,
                                                                   global_step,
                                                                   params['learning_rate.cosine_decay']['steps'],
                                                                   m_mul=params['learning_rate.cosine_decay']['m_mul'],
                                                                   alpha=params['learning_rate.cosine_decay']['alpha'])
                elif params['learning_rate.exponential_decay'] is not None:
                    tf.logging.info('learning_rate: exponential_decay')
                    learning_rate = tf.train.exponential_decay(lr,
                                                               global_step,
                                                               params['learning_rate.exponential_decay']['decay_steps'],
                                                               params['learning_rate.exponential_decay']['decay'],
                                                               staircase=True)
                else:
                    tf.logging.info('learning_rate: constant')
                    learning_rate = tf.constant(lr)
                tf.summary.scalar('learning_rate', learning_rate)

                map_iou = metric.map_iou(predicted_probs, labels,
                                         thresholds=params['map_iou_thresholds'],
                                         pred_thresh=params['map_iou_predthresh'])
                tf.summary.scalar('map_iou', map_iou)

                logging_hook = tf.train.LoggingTensorHook({'map_iou': map_iou, 'learning_rate': learning_rate},
                                                          every_n_iter=10)

                adam_epsilon = 1e-8 if params['adam_epsilon'] is None else params['adam_epsilon']
                tf.logging.info(f'ADAM EPSILON: {adam_epsilon}')
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)

                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    grads_and_vars = optimizer.compute_gradients(loss)
                    if params['clip_grad_norm'] is not None:
                        grads_and_vars = self.clip_gradient_norms(grads_and_vars, params['clip_grad_norm'])
                    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                if params['ema_decay'] is not None:
                    model_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                    ema = tf.train.ExponentialMovingAverage(decay=params['ema_decay'])
                    with tf.control_dependencies([train_op]):
                        train_op = ema.apply(model_vars)

                spec = tf.estimator.EstimatorSpec(mode,
                                                  predictions=predictions,
                                                  loss=loss,
                                                  train_op=train_op,
                                                  training_hooks=[logging_hook])
        return spec
