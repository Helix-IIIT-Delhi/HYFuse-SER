import math
import tensorflow as tf

def _softplus_inv_scalar(y: float) -> float:
    # Numerically stable inverse of softplus for a positive scalar y
    # y = log(1 + exp(x)) => x = log(exp(y) - 1)
    y = max(float(y), 1e-8)
    return math.log(math.expm1(y))

@tf.keras.utils.register_keras_serializable(package="hypolayers")
class Numerics(tf.keras.layers.Layer):
    """Utility base layer providing curvature management and safe numerics.

    Args:
        curvature: initial positive curvature (>0)
        eps: numerical epsilon for safe divisions/norms
        trainable_curvature: if True, learn curvature parameter
        curvature_param: "exp" (c = min_c + exp(raw)) or "softplus" (c = min_c + softplus(raw))
        min_curvature: lower bound to keep c > 0 and radius finite
        curvature_regularizer: optional Keras regularizer for the curvature parameter
    """
    def __init__(self,
                 curvature: float = 1.0,
                 eps: float = 1e-7,
                 trainable_curvature: bool = False,
                 curvature_param: str = "exp",
                 min_curvature: float = 1e-4,
                 curvature_regularizer=None,
                 **kwargs):
        super().__init__(**kwargs)
        if curvature <= 0:
            raise ValueError("curvature must be > 0")
        if curvature_param not in ("exp", "softplus"):
            raise ValueError("curvature_param must be 'exp' or 'softplus'")
        self.curvature = float(curvature)
        self.eps = float(eps)
        self.trainable_curvature = bool(trainable_curvature)
        self.curvature_param = str(curvature_param)
        self.min_curvature = float(min_curvature)
        self.curvature_regularizer = tf.keras.regularizers.get(curvature_regularizer)

        self._c_const = None
        self._raw_c = None

    def build(self, input_shape):
        if self.trainable_curvature:
            # initialize raw parameter so that resulting c ≈ self.curvature
            target = max(self.curvature - self.min_curvature, 1e-8)
            if self.curvature_param == "exp":
                raw_init = math.log(target)
            else:
                raw_init = _softplus_inv_scalar(target)
            self._raw_c = self.add_weight(
                name="raw_c",
                shape=(),
                initializer=tf.keras.initializers.Constant(raw_init),
                trainable=True,
                regularizer=self.curvature_regularizer,
                dtype=tf.float32,
            )
        else:
            self._c_const = tf.constant(self.curvature, dtype=tf.float32)
        super().build(input_shape)

    def _curvature_tensor(self, dtype):
        if self.trainable_curvature:
            if self.curvature_param == "exp":
                c = self.min_curvature + tf.exp(self._raw_c)
            else:
                c = self.min_curvature + tf.nn.softplus(self._raw_c)
        else:
            c = self._c_const
        return tf.cast(c, dtype)

    def _cast_consts(self, x):
        dtype = x.dtype
        c = self._curvature_tensor(dtype)
        eps = tf.cast(self.eps, dtype)
        sqrt_c = tf.sqrt(c)
        return c, sqrt_c, eps

    def norm_safe(self, x, eps):
        return tf.maximum(tf.linalg.norm(x, axis=-1, keepdims=True), eps)

    def project_to_ball(self, x, c, eps):
        # Ensure ||x|| < 1/sqrt(c) with a small margin
        r = tf.linalg.norm(x, axis=-1, keepdims=True)
        max_r = (1.0 - 1e-6) / tf.sqrt(c)
        scale = tf.minimum(1.0, max_r / tf.maximum(r, eps))
        return x * scale

    def get_config(self):
        base = super().get_config()
        base.update({
            "curvature": self.curvature,
            "eps": self.eps,
            "trainable_curvature": self.trainable_curvature,
            "curvature_param": self.curvature_param,
            "min_curvature": self.min_curvature,
            "curvature_regularizer": tf.keras.regularizers.serialize(self.curvature_regularizer),
        })
        return base
