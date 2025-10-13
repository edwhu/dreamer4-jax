from functools import partial
import jax
import jax.numpy as jnp
import optax
from flax.core import freeze, unfreeze, FrozenDict
from models import Encoder, Dynamics
from data import make_iterator
import imageio
from pathlib import Path
from time import time
from utils import temporal_patchify, temporal_unpatchify, make_state, make_manager, try_restore, maybe_save
from einops import rearrange

def _with_params(variables, new_params):
    """Replace the 'params' collection in a Flax variables PyTree."""
    d = unfreeze(variables) if isinstance(variables, FrozenDict) else dict(variables)
    d["params"] = new_params
    return freeze(d)

@partial(
    jax.jit,
    static_argnames=("shape_bt", "k_max",),
)
def sample_shortcut_indices(rng, shape_bt, k_max: int, *, dtype=jnp.float32):
    """
    Returns:
      d:         (B,T) float32   step sizes in {1/K | K in {1,2,4,...,k_max}}
      tau:       (B,T) float32   signal levels in {0, d, 2d, ..., 1-d}
      step_idx:  (B,T) int32     e = log2(K)  in {0, ..., log2(k_max)}
      tau_idx:   (B,T) int32     m = tau * k_max in {0, ..., k_max-1}
    """
    B, T = shape_bt
    emax = jnp.log2(k_max).astype(jnp.int32)
    rng_step, rng_tau = jax.random.split(rng)

    # Sample exponent e -> K = 2^e, d = 1/K
    step_idx = jax.random.randint(rng_step, (B, T), 0, emax + 1, dtype=jnp.int32)  # e
    K = (1 << step_idx)                              # (B,T) int32
    d = (1.0 / K.astype(dtype))                      # (B,T) float32

    # Sample j ∈ {0,...,K-1} -> tau = j/K
    u = jax.random.uniform(rng_tau, (B, T), dtype=dtype)  # [0,1)
    j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)  # (B,T) int32 in [0, K-1]
    tau = j_idx.astype(dtype) / K.astype(dtype)      # (B,T) float32

    # Map tau to a *global* K_max-grid index m = tau * K_max = j * (K_max // K)
    # Do it in integers to avoid rounding error:
    tau_idx = j_idx * (k_max // K)                   # (B,T) int32 in [0, k_max-1]

    return d, tau, step_idx, tau_idx

def pack_bottleneck_to_spatial(z_btLd, *, n_s: int, k: int):
    """
    (B,T,N_b,D_b) -> (B,T,S_z, D_z_pre) by merging k tokens along N_b into channels.
    Requires: N_b == n_s * k  (e.g., 512 -> 256 with k=2).
    """
    return rearrange(z_btLd, 'b t (n_s k) d -> b t n_s (k d)', n_s=n_s, k=k)


def init_models(rng, encoder, dynamics, patch_tokens, B, T, enc_n_latents, enc_d_bottleneck, packing_factor, num_spatial_tokens):
    rng, params_rng, mae_rng, dropout_rng = jax.random.split(rng, 4)

    enc_vars = encoder.init(
        {"params": params_rng, "mae": mae_rng, "dropout": dropout_rng},
        patch_tokens, deterministic=True
    )
    fake_enc_z = jnp.ones((B, T, enc_n_latents, enc_d_bottleneck), dtype=jnp.float32)
    fake_packed_z = pack_bottleneck_to_spatial(fake_enc_z, n_s=num_spatial_tokens, k=packing_factor)
    fake_actions = jnp.ones((B, T), dtype=jnp.int32)
    fake_signals = jnp.full((B, T), 0.0, dtype=jnp.float32)
    fake_steps = jnp.full((B, T), 1/256, dtype=jnp.float32)
    fake_step_idxs = jnp.full((B, T), 0, dtype=jnp.int32)
    fake_signal_idxs = jnp.full((B, T), 0, dtype=jnp.int32)
    dynamics_vars = dynamics.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_actions,
        fake_step_idxs,
        fake_signal_idxs,
        fake_packed_z,
    )
    return rng, enc_vars, dynamics_vars



@partial(
    jax.jit,
    static_argnames=(
        "encoder","dynamics","tx","patch","H","W","C",
        "n_s","k_max","normalize_loss","packing_factor",
    ),
)
def train_step(
    encoder: Encoder, dynamics: Dynamics, tx,
    params, opt_state,
    enc_vars, dynamics_vars,
    frames, actions,
    *,
    patch, H, W, C,
    n_s: int, k_max: int, packing_factor: int,
    normalize_loss: bool = True,
    master_key: jnp.ndarray, step: int,
):
    # ---------- precompute (param-free) ----------
    patches_btnd = temporal_patchify(frames, patch)  # (B,T,Np,Dp)

    step_key  = jax.random.fold_in(master_key, step)
    enc_key, noise_key, idx_key, drop_key = jax.random.split(step_key, 4)

    # Frozen encoder forward
    z_b_clean, _ = encoder.apply(
        enc_vars, patches_btnd, rngs={"mae": enc_key}, deterministic=True
    )
    B, T, _, _ = z_b_clean.shape

    # Pack, sample indices, corrupt
    z1 = pack_bottleneck_to_spatial(z_b_clean, n_s=n_s, k=packing_factor)  # (B,T,Sz,Dz)
    step_values, signal_values, step_idxs, signal_idxs = sample_shortcut_indices(idx_key, (B, T), k_max)

    z0      = jax.random.normal(noise_key, z1.shape, dtype=z1.dtype)
    z_tilde = (1 - signal_values) * z0 + signal_values * z1

    # prepend zero action so t=0 uses a null action
    actions = jnp.concatenate([jnp.zeros((B, 1), dtype=actions.dtype), actions], axis=1)

    # constants/derived indices that don't depend on params
    max_step_idx         = jnp.log2(k_max).astype(jnp.int32)
    half_step_values     = step_values / 2
    half_step_idxs       = jnp.clip(step_idxs + 1, 0, max_step_idx)
    new_signal_values    = signal_values + half_step_values
    half_step_signal_idxs= (k_max * half_step_values).astype(jnp.int32)
    new_signal_idxs      = signal_idxs + half_step_signal_idxs
    step_mask            = step_idxs == max_step_idx
    ramp_weight          = 0.9 * signal_values + 0.1  # (B,T)

    # ---------- loss closure (param-dependent only) ----------
    def loss_and_metrics_with_params(p):
        dyn_vars_local = _with_params(dynamics_vars, p)

        # current step
        z1_hat = dynamics.apply(
            dyn_vars_local, actions, step_idxs, signal_idxs, z_tilde,
            rngs={"dropout": drop_key}, deterministic=False,
        )

        # first half-step
        b1 = (dynamics.apply(
            dyn_vars_local, actions, half_step_idxs, signal_idxs, z_tilde,
            rngs={"dropout": drop_key}, deterministic=False,
        ) - z_tilde) / (1 - signal_values)

        z_mid = z_tilde + b1 * half_step_values

        # second half-step (advance signal)
        b2 = (dynamics.apply(
            dyn_vars_local, actions, half_step_idxs, new_signal_idxs, z_mid,
            rngs={"dropout": drop_key}, deterministic=False,
        ) - z_mid) / (1 - new_signal_values)

        # targets & preds
        b_targets = jax.lax.stop_gradient(b1 + b2) / 2.0
        b_preds   = (z1_hat - z_tilde) / (1 - signal_values)

        # per-sample losses (B,T)
        b_loss_per    = (1 - signal_values) ** 2 * jnp.mean((b_preds - b_targets) ** 2, axis=(2, 3))
        flow_loss_per = jnp.mean((z1_hat - z_tilde) ** 2, axis=(2, 3))

        # choose loss per (B,T)
        loss_per = jnp.where(step_mask, flow_loss_per, b_loss_per)

        # ramp weighting → scalar loss
        loss = jnp.mean(loss_per * ramp_weight)

        aux = {
            "loss": loss,
            "flow_loss": jnp.mean(flow_loss_per * step_mask),
            "weighted_flow_loss": jnp.mean(flow_loss_per * step_mask * ramp_weight),
            "bootstrap_loss": jnp.mean(b_loss_per * ~step_mask),
            "weighted_bootstrap_loss": jnp.mean(b_loss_per * ~step_mask * ramp_weight),
        }
        return loss, aux

    (loss_val, aux_out), grads = jax.value_and_grad(
        loss_and_metrics_with_params, has_aux=True
    )(params)

    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_dynamics_vars = _with_params(dynamics_vars, new_params)

    return new_params, opt_state, new_dynamics_vars, aux_out


if __name__ == "__main__":
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = "test_dynamics"
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)


    rng = jax.random.PRNGKey(0)
    # dataset parameters
    B, T, H, W, C = 8, 64, 32, 32, 3
    pixels_per_step = 2 # how many pixels the agent moves per step
    size_min = 6 # minimum size of the square
    size_max = 14 # maximum size of the square
    hold_min = 4 # how long the agent holds a direction for
    hold_max = 9 # how long the agent holds a direction for

    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = 256

    # data
    next_batch = make_iterator(B, T, H, W, C, pixels_per_step, size_min, size_max, hold_min, hold_max)

    rng, batch_rng = jax.random.split(rng)
    rng, (frames, actions) = next_batch(rng) 

    # models 
    enc_n_latents, enc_d_bottleneck = 16, 32
    enc_kwargs = {
        "d_model": 64, "n_latents": enc_n_latents, "n_patches": num_patches, "n_heads": 4, "depth": 8, "dropout": 0.0,
        "d_bottleneck": enc_d_bottleneck, "mae_p_min": 0.1, "mae_p_max": 0.1, "time_every": 4,
    }
    packing_factor = 2
    n_s = enc_n_latents // packing_factor
    dynamics_kwargs = {
        "d_model": 128,
        "n_s": n_s,
        "d_spatial": enc_d_bottleneck * packing_factor,
        "d_bottleneck": enc_d_bottleneck,
        "k_max": k_max,
        "n_r": 10,
        "n_heads": 4,
        "depth": 4,
        "dropout": 0.0
    }

    encoder = Encoder(**enc_kwargs)
    dynamics = Dynamics(**dynamics_kwargs)

    init_patches = temporal_patchify(frames, patch)
    rng, enc_vars, dynamics_vars = init_models(rng, encoder, dynamics, init_patches, B, T, enc_n_latents, enc_d_bottleneck, packing_factor, n_s)

    params = dynamics_vars["params"]
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)
    max_steps = 1_000_000

    # ---------- ORBAX: manager + (optional) restore ----------
    ckpt_dir = run_dir / "checkpoints"
    mngr = make_manager(ckpt_dir, max_to_keep=5, save_interval_steps=10_000)

    for step in range(max_steps):
        data_start_t = time()
        rng, (frames, actions) = next_batch(rng)
        data_t = time() - data_start_t
        train_start_t = time()
        rng, master_key = jax.random.split(rng)
        params, opt_state, dynamics_vars, aux = train_step(
            encoder, dynamics, tx, params, opt_state, enc_vars, dynamics_vars, frames, actions,
            patch=patch, H=H, W=W, C=C, master_key=master_key, step=step, packing_factor=packing_factor, n_s=n_s, k_max=k_max,
        )
        train_t = time() - train_start_t
        break