import tensorflow as tf
from .utils import Numerics

@tf.keras.utils.register_keras_serializable(package="hypolayers")
class LogMap0(Numerics):
    """Logarithmic map at the origin for the Poincaré ball of curvature c.

    log_0(x) = (1/√c) * artanh(√c ||x||) * x / ||x||
    """
    def call(self, x):
        c, sqrt_c, eps = self._cast_consts(x)
        r = self.norm_safe(x, eps)
        t = tf.minimum(sqrt_c * r, tf.constant(1.0 - 1e-6, dtype=x.dtype))
        factor = (1.0 / sqrt_c) * tf.atanh(t) / r
        return factor * x

# Functional wrapper
def log_map0(x, curvature=1.0, eps=1e-7, trainable_curvature=False,
             curvature_param="exp", min_curvature=1e-4, curvature_regularizer=None):
    return LogMap0(curvature=curvature, eps=eps, trainable_curvature=trainable_curvature,
                   curvature_param=curvature_param, min_curvature=min_curvature,
                   curvature_regularizer=curvature_regularizer)(x)
