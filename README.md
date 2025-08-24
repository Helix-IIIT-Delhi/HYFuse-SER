# hypolayers (v1.0.0)

Hyperbolic layers for TensorFlow/Keras (Poincaré ball) with optional **learnable curvature**.

- Use **fixed curvature** (default) or set `trainable_curvature=True` to learn `c`.
- Safe parameterizations: `curvature_param="exp"` (default) or `"softplus"`.
- Enforces `c > 0` via a lower bound `min_curvature`.
- Keras-serializable; supports `model.save()` / `load_model()`.

## Install (local)
```bash
pip install -e .



```bash
from hypolayers.hypo import ExpMap0
from hypolayers.mobious import MobiusAdd
from hypolayers.euclidien import LogMap0

exp0 = ExpMap0(curvature=1.0)
mob  = MobiusAdd(curvature=1.0)
log0 = LogMap0(curvature=1.0)
```

```bash 

from hypolayers.hypo import ExpMap0
from hypolayers.mobius import MobiusAdd
from hypolayers.euclidean import LogMap0

exp0 = ExpMap0(curvature=0.8, trainable_curvature=True, curvature_param="exp")
mob  = MobiusAdd(curvature=0.8, trainable_curvature=True)
log0 = LogMap0(curvature=0.8, trainable_curvature=True)
```

```
from hypolayers import exp_map0, mobius_add, log_map0
z_h = mobius_add(exp_map0(emb_a), exp_map0(emb_b))
z_e = log_map0(z_h)
```