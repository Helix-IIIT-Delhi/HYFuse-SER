import tensorflow as tf
from .utils import Numerics

@tf.keras.utils.register_keras_serializable(package="hypolayers")
class ExpMap0(Numerics):
    """Exponential map at the origin for the Poincaré ball of curvature c.

    exp_0(v) = tanh(√c ||v||) / (√c ||v||) * v
    """
    def call(self, v):
        c, sqrt_c, eps = self._cast_consts(v)
        r = self.norm_safe(v, eps)
        denom = tf.maximum(sqrt_c * r, eps)
        factor = tf.math.tanh(sqrt_c * r) / denom
        y = factor * v
        return self.project_to_ball(y, c, eps)

# Functional wrapper
def exp_map0(v, curvature=1.0, eps=1e-7, trainable_curvature=False,
             curvature_param="exp", min_curvature=1e-4, curvature_regularizer=None):
    return ExpMap0(curvature=curvature, eps=eps, trainable_curvature=trainable_curvature,
                   curvature_param=curvature_param, min_curvature=min_curvature,
                   curvature_regularizer=curvature_regularizer)(v)
