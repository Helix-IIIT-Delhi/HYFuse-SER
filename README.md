## Fixed curvature (default)

```
from hypolayers.hypo import ExpMap0           # Euclidean -> Hyperbolic
from hypolayers.mobious import MobiusAdd      # Möbius addition (alias spelling)
from hypolayers.euclidien import LogMap0      # Hyperbolic -> Euclidean (alias spelling)

exp0 = ExpMap0(curvature=1.0)
mob  = MobiusAdd(curvature=1.0)
log0 = LogMap0(curvature=1.0)

h_a = exp0(emb_a)          # [batch, d]
h_b = exp0(emb_b)
h_f = mob([h_a, h_b])      # hyperbolic fusion
z   = log0(h_f)            # back to Euclidean → Dense/Softmax...
```

## Learnable curvature

```
from hypolayers.hypo import ExpMap0
from hypolayers.mobius import MobiusAdd
from hypolayers.euclidean import LogMap0

exp0 = ExpMap0(curvature=0.8, trainable_curvature=True, curvature_param="exp")  # or "softplus"
mob  = MobiusAdd(curvature=0.8, trainable_curvature=True)
log0 = LogMap0(curvature=0.8, trainable_curvature=True)
```

## Functional wrappers

```
from hypolayers import exp_map0, mobius_add, log_map0

h_a = exp_map0(emb_a, curvature=1.0)
h_b = exp_map0(emb_b, curvature=1.0)
h_f = mobius_add(h_a, h_b, curvature=1.0)
z   = log_map0(h_f, curvature=1.0)
```

## Example: dual-modality CNN with hyperbolic fusion

```
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from hypolayers.hypo import ExpMap0
from hypolayers.mobious import MobiusAdd
from hypolayers.euclidien import LogMap0

def build_hyperbolic_cnn_model(input_shape1, input_shape2, output_dim,
                               embedding_dim=128, curvature=1.0, learn_c=False):

    a_in = Input(shape=input_shape1, name="modality_a")  # e.g. (128, 1)
    b_in = Input(shape=input_shape2, name="modality_b")  # e.g. (128, 1)

    # Branch A
    xa = Conv1D(64, 3, activation='relu')(a_in)
    xa = MaxPooling1D(2)(xa); xa = BatchNormalization()(xa)
    xa = Conv1D(32, 3, activation='relu')(xa)
    xa = MaxPooling1D(2)(xa)
    xa = Flatten()(xa)
    emb_a = Dense(embedding_dim)(xa)

    # Branch B
    xb = Conv1D(64, 3, activation='relu')(b_in)
    xb = MaxPooling1D(2)(xb); xb = BatchNormalization()(xb)
    xb = Conv1D(32, 3, activation='relu')(xb)
    xb = MaxPooling1D(2)(xb)
    xb = Flatten()(xb)
    emb_b = Dense(embedding_dim)(xb)

    # Hyperbolic ops
    exp0 = ExpMap0(curvature=curvature, trainable_curvature=learn_c)
    mob  = MobiusAdd(curvature=curvature, trainable_curvature=learn_c)
    log0 = LogMap0(curvature=curvature, trainable_curvature=learn_c)

    h_a = exp0(emb_a)
    h_b = exp0(emb_b)            # same exp0 instance → shared curvature
    h_f = mob([h_a, h_b])
    z   = log0(h_f)

    z = Dense(64, activation='relu')(z)
    z = Dropout(0.5)(z)
    out = Dense(output_dim, activation='softmax')(z)

    return Model([a_in, b_in], out, name="HyperbolicFusionModel")
```