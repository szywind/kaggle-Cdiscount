import keras.backend as K
from keras.losses import categorical_crossentropy
import tensorflow as tf

# def focus_loss(y_true, y_pred, gamma=0.):
#     y_pred /= K.sum(y_pred, axis=len(y_pred.get_shape())-1, keepdims=True)

#     y_true = K.flatten(y_true)
#     y_pred = K.flatten(y_pred)
#
#
#     loss = y_true*K.log(y_pred+K.epsilon())*(1-y_pred+K.epsilon())**gamma + \
#            (1-y_true)*K.log(1-y_pred+K.epsilon())*(y_pred+K.epsilon())**gamma
#     return -K.mean(loss)



def focus_loss(y_true, y_pred, gamma=2.):
    y_pred /= K.sum(y_pred, axis=-1, keepdims=True)

    epsilon = K.epsilon()

    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)

    loss = K.pow(1.0 - y_pred, gamma)

    loss = - K.sum(loss * y_true * K.log(y_pred), axis=-1)
    return loss



# def categorical_crossentropy(output, y_true, from_logits=False):
#     """Categorical crossentropy between an output tensor and a target tensor.
#
#     # Arguments
#         output: A tensor resulting from a softmax
#             (unless `from_logits` is True, in which
#             case `output` is expected to be the logits).
#         target: A tensor of the same shape as `output`.
#         from_logits: Boolean, whether `output` is the
#             result of a softmax, or is a tensor of logits.
#
#     # Returns
#         Output tensor.
#     """
#     # Note: tf.nn.softmax_cross_entropy_with_logits
#     # expects logits, Keras expects probabilities.
#     if not from_logits:
#         # scale preds so that the class probas of each sample sum to 1
#         output /= tf.reduce_sum(output,
#                                 axis=len(output.get_shape()) - 1,
#                                 keep_dims=True)
#         # manual computation of crossentropy
#         epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
#         output = tf.clip_by_value(output, epsilon, 1. - epsilon)
#         return - tf.reduce_sum(target * tf.log(output),
#                                axis=len(output.get_shape()) - 1)
#     else:
#         return tf.nn.softmax_cross_entropy_with_logits(labels=target,
#                                                        logits=output)