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

def sample_shortcut_indices(rng, shape_bt, k_max: int):
    """
    Returns:
      step_idx:   (B,T) int32 in [0, log2(k_max)]   where K=2^{step_idx}
      signal_idx: (B,T) int32 in [0, K-1]           (grid index within the chosen K)
      tau:        (B,T) float32 in {0/K, 1/K, ..., (K-1)/K}
    """
    B, T = shape_bt
    max_step = int(jnp.log2(k_max))
    rng_step, rng_tau = jax.random.split(rng)

    step_idx = jax.random.randint(rng_step, (B, T), minval=0, maxval=max_step + 1, dtype=jnp.int32)  # k ~ U{0..max}
    K = (1 << step_idx).astype(jnp.int32)  # K = 2^k

    # Per-element tau index: uniform integer in [0, K-1]
    u = jax.random.uniform(rng_tau, (B, T), dtype=jnp.float32)  # in [0,1)
    signal_idx = jnp.minimum((u * K.astype(jnp.float32)).astype(jnp.int32), K - 1)

    tau = signal_idx.astype(jnp.float32) / K.astype(jnp.float32)  # tau ∈ {0/K, ..., (K-1)/K}
    return step_idx, signal_idx, tau

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
    fake_signal_idx = jnp.ones((B, T), dtype=jnp.int32)
    fake_step_idx = jnp.ones((B, T), dtype=jnp.int32)
    dynamics_vars = dynamics.init(
        {"params": params_rng, "dropout": dropout_rng},
        fake_packed_z,
        fake_actions,
        fake_signal_idx,
        fake_step_idx,
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
    params, opt_state,               # params = dynamics params ONLY
    enc_vars, dynamics_vars,         # frozen encoder vars; dynamics vars for non-param cols
    frames, actions,        # batch inputs
    *,
    patch, H, W, C,
    n_s: int, k_max: int, packing_factor:int,
    normalize_loss: bool = True,
    master_key: jnp.ndarray, step: int,
):
    """
    One training step:
      1) Tokenize frames with FROZEN encoder → bottleneck z_clean: (B,T,N_b,D_b)
      2) Sample (step_idx, signal_idx, tau) per (B,T)
      3) Corrupt z_clean in bottleneck space: z_tilde = tau * z_clean + (1-tau) * z0
      4) Run dynamics(enc_z=z_tilde, actions, signal_idx, step_idx) → z_hat (B,T,S_z,D_model)
      5) Build clean target in projected space with SAME pack + enc_z_proj weights → z_clean_proj
      6) Loss = mean squared error(z_hat, z_clean_proj)
      7) Optax update on dynamics params only
    Returns:
      new_params, new_opt_state, new_dynamics_vars, metrics
    """
    # --- 1) Preprocess frames → patch tokens for the encoder
    patches_btnd = temporal_patchify(frames, patch)  # (B,T,N_p,D_p)

    # Per-step RNGs
    step_key  = jax.random.fold_in(master_key, step)
    enc_key, noise_key, idx_key, drop_key = jax.random.split(step_key, 4)

    # --- 2) Get FROZEN encoder bottleneck z_clean: (B,T,N_b,D_b)
    # Run encoder deterministically (no MAE masking); if your encoder ignores rngs when deterministic=True,
    # you can pass empty rngs; otherwise pass {"mae": enc_key} safely (it shouldn't mask in eval).
    z_clean, _ = encoder.apply(
        enc_vars, patches_btnd, rngs={"mae": enc_key}, deterministic=True
    )

    B, T, N_b, D_b = z_clean.shape
    z_packed_clean = pack_bottleneck_to_spatial(z_clean, n_s=n_s, k=packing_factor)
    import ipdb; ipdb.set_trace()

    # TODO: implement shortcut forcing.

    # # --- 3) Sample shortcut indices + corruption in bottleneck space
    # step_idx_bt, signal_idx_bt, tau_bt = sample_shortcut_indices(idx_key, (B, T), k_max)
    # z0_btLd = jax.random.normal(noise_key, z_clean.shape, dtype=z_clean.dtype)
    # mix = tau_bt[..., None, None]        # (B,T,1,1)
    # z_tilde_btLd = mix * z_clean + (1.0 - mix) * z0_btLd

    # # --- 4) Forward dynamics on z_tilde
    # dyn_vars_in = _with_params(dynamics_vars, params)
    # z_hat_btSzDz = dynamics.apply(
    #     dyn_vars_in,
    #     z_tilde_btLd,          # enc_z (noisy)
    #     actions_bt,            # (B,T) ints
    #     signal_idx_bt,         # (B,T) ints
    #     step_idx_bt,           # (B,T) ints
    #     rngs={"dropout": drop_key},
    #     deterministic=False,
    # )  # (B,T,S_z,d_model)

    # # --- 5) Build CLEAN target in projected space with SAME weights
    # # Pack: (B,T,N_b,D_b) -> (B,T,S_z, D_pre = 2*D_b)
    # z_clean_pack = pack_bottleneck_to_spatial(z_clean, n_s=n_s, k=2)  # (B,T,S_z, 2*D_b)

    # # Project using the CURRENT enc_z_proj weights from dynamics params
    # dparams = params  # dynamics params
    # if z_clean_pack.shape[-1] != dparams["enc_z_proj"]["kernel"].shape[0]:
    #     # If your enc_z_proj expects different in-dim you likely changed k; assert or adapt.
    #     raise ValueError("enc_z_proj input dim mismatch. Check n_s and k.")
    # W = dparams["enc_z_proj"]["kernel"]       # (D_pre, d_model)
    # b = dparams["enc_z_proj"]["bias"]         # (d_model,)
    # z_clean_proj = jnp.einsum('...d,df->...f', z_clean_pack, W) + b  # (B,T,S_z,d_model)

    # # --- 6) x-pred loss in projected space
    # diff = z_hat_btSzDz - z_clean_proj
    # mse = jnp.sum(diff * diff)
    # denom = diff.size if normalize_loss else 1.0
    # loss = mse / denom

    # metrics = {
    #     "loss": loss,
    #     "mse": mse / denom,
    #     "K_mean": jnp.mean((1 << step_idx_bt).astype(jnp.float32)),
    #     "tau_mean": jnp.mean(tau_bt),
    # }

    # # --- 7) Backprop on dynamics params only
    # def loss_only(p):
    #     dyn_vars_local = _with_params(dynamics_vars, p)
    #     # Recompute z_hat with *p*; reuse z_tilde and indices (pure JAX values)
    #     z_hat = dynamics.apply(
    #         dyn_vars_local,
    #         z_tilde_btLd, actions_bt, signal_idx_bt, step_idx_bt,
    #         rngs={"dropout": drop_key},
    #         deterministic=False,
    #     )
    #     # Target must be recomputed if enc_z_proj lives in params (it does).
    #     W = p["enc_z_proj"]["kernel"]
    #     b = p["enc_z_proj"]["bias"]
    #     z_clean_proj_local = jnp.einsum('...d,df->...f', z_clean_pack, W) + b
    #     d = z_hat - z_clean_proj_local
    #     m = jnp.sum(d * d)
    #     return m / denom

    # grads = jax.grad(loss_only)(params)
    # updates, opt_state = tx.update(grads, opt_state, params)
    # new_params = optax.apply_updates(params, updates)

    # # No non-param state in dynamics by default; return as-is
    # new_dyn_vars = _with_params(dynamics_vars, new_params)

    return new_params, opt_state, new_dyn_vars, metrics


if __name__ == "__main__":
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = "test_dynamics"
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)


    rng = jax.random.PRNGKey(0)
    # dataset parameters
    B, T, H, W, C = 64, 4, 32, 32, 3
    pixels_per_step = 2 # how many pixels the agent moves per step
    size_min = 6 # minimum size of the square
    size_max = 14 # maximum size of the square
    hold_min = 4 # how long the agent holds a direction for
    hold_max = 9 # how long the agent holds a direction for

    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = 8

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