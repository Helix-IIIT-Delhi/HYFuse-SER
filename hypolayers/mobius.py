import tensorflow as tf
from .utils import Numerics

@tf.keras.utils.register_keras_serializable(package="hypolayers")
class MobiusAdd(Numerics):
    """Möbius addition at the origin in the Poincaré ball of curvature c."""
    def call(self, inputs):
        x, y = inputs
        c, _, eps = self._cast_consts(x)
        x2 = c * tf.reduce_sum(x * x, axis=-1, keepdims=True)
        y2 = c * tf.reduce_sum(y * y, axis=-1, keepdims=True)
        xy = c * tf.reduce_sum(x * y, axis=-1, keepdims=True)
        num = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
        den = 1.0 + 2.0 * xy + x2 * y2
        z = num / tf.maximum(den, eps)
        return self.project_to_ball(z, c, eps)

# Functional wrapper
def mobius_add(x, y, curvature=1.0, eps=1e-7, trainable_curvature=False,
               curvature_param="exp", min_curvature=1e-4, curvature_regularizer=None):
    return MobiusAdd(curvature=curvature, eps=eps, trainable_curvature=trainable_curvature,
                     curvature_param=curvature_param, min_curvature=min_curvature,
                     curvature_regularizer=curvature_regularizer)([x, y])
