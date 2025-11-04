from functools import partial
import jax
import jax.numpy as jnp
import optax
from models import Encoder, Decoder, Dynamics
from data import make_iterator
from pathlib import Path
from time import time
from utils import pack_mae_params, temporal_patchify, make_state, make_manager, try_restore, maybe_save, with_params, pack_bottleneck_to_spatial
from einops import rearrange
import orbax.checkpoint as ocp


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



@partial(jax.jit, static_argnames=("shape_bt","k_max",))
def _sample_tau_for_step(rng, shape_bt, k_max:int, step_idx:jnp.ndarray, *, dtype=jnp.float32):
    """Given per-element step_idx (e = log2 K), sample tau uniformly on that step's grid."""
    B, T = shape_bt
    rng_tau = rng
    K = (1 << step_idx)                             # (B,T)
    u = jax.random.uniform(rng_tau, (B, T), dtype=dtype)
    j_idx = jnp.floor(u * K.astype(dtype)).astype(jnp.int32)   # 0..K-1
    tau = j_idx.astype(dtype) / K.astype(dtype)                 # (B,T)
    tau_idx = j_idx * (k_max // K)                              # global grid index
    return tau, tau_idx

@partial(jax.jit, static_argnames=("shape_bt","k_max",))
def _sample_step_excluding_dmin(rng, shape_bt, k_max:int):
    """Sample step exponents e in [0, emax) (exclude d_min), return (d, step_idx)."""
    B, T = shape_bt
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jax.random.randint(rng, (B, T), 0, emax, dtype=jnp.int32)  # exclude emax
    d = 1.0 / (1 << step_idx).astype(jnp.float32)
    return d, step_idx


@partial(
    jax.jit,
    static_argnames=("encoder","dynamics","tx","patch",
                     "n_s","k_max","packing_factor","B","T","B_self"),
)
def train_step_efficient(
    encoder, dynamics, tx,
    params, opt_state,
    enc_vars, dynamics_vars,
    frames, actions,
    *,
    patch: int,
    B: int, T: int, B_self: int,            # assume 0 < B_self < B (deterministic split)
    n_s: int, k_max: int, packing_factor: int,
    master_key: jnp.ndarray, step: int,
):
    """
    Deterministic two-branch training:
      - first B_emp rows: empirical flow at smallest step d_min = 1/k_max
      - last  B_self rows: self-consistency with d > d_min
    One fused main forward (emp + self), then two half-step forwards only for self rows.

    Notation:
      sigma ∈ [0,1]  : signal level
      d ∈ {1, 1/2, 1/4, ..., 1/k_max} : step size (delta)
      z_tilde = (1 - sigma) z0 + sigma z1
      b'  = (f(z_tilde, sigma, d/2) - z_tilde) / (1 - sigma)
      z'  = z_tilde + b' * (d/2)
      b'' = (f(z', sigma + d/2, d/2) - z') / (1 - (sigma + d/2))
      vhat_sigma = (f(z_tilde, sigma, d) - z_tilde) / (1 - sigma)
      L_flow = || f(z_tilde, sigma, d_min) - z1 ||^2
      L_boot = (1 - sigma)^2 * || vhat_sigma - stopgrad((b' + b'')/2) ||^2
      w(sigma) = 0.9 * sigma + 0.1
    """
    # ---------- Param-free precompute ----------
    patches_btnd = temporal_patchify(frames, patch)  # (B,T,Np,Dp)

    # RNGs: encoder MAE, sigma sampling, self step sampling, corruption, dropout
    step_key = jax.random.fold_in(master_key, step)
    enc_key, key_sigma_full, key_step_self, key_noise_full, drop_key = jax.random.split(step_key, 5)

    # Frozen encoder → bottleneck → spatial tokens (clean target z1)
    z_bottleneck, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": enc_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_bottleneck, n_s=n_s, k=packing_factor)  # (B,T,Sz,Dz)

    # Deterministic batch split
    B_emp  = B - B_self
    actions_full = actions
    emax = jnp.log2(k_max).astype(jnp.int32)  # exponent index for d_min

    # --- Step indices (encode d) ---
    step_idx_emp  = jnp.full((B_emp,  T), emax, dtype=jnp.int32)             # d = d_min for empirical rows
    d_self, step_idx_self = _sample_step_excluding_dmin(key_step_self, (B_self, T), k_max)  # d > d_min for self rows
    step_idx_full = jnp.concatenate([step_idx_emp, step_idx_self], axis=0)   # (B,T)

    # --- Signal levels on each row's grid (one call for whole batch) ---
    sigma_full, sigma_idx_full = _sample_tau_for_step(key_sigma_full, (B, T), k_max, step_idx_full)
    sigma_emp   = sigma_full[:B_emp]                   # (B_emp,T)
    sigma_self  = sigma_full[B_emp:]                   # (B_self,T)
    sigma_idx_self = sigma_idx_full[B_emp:]            # (B_self,T)

    # --- Corrupt inputs: z_tilde = (1 - sigma) z0 + sigma z1 (one draw for all rows) ---
    z0_full      = jax.random.normal(key_noise_full, z1.shape, dtype=z1.dtype)
    z_tilde_full = (1.0 - sigma_full)[...,None,None] * z0_full + sigma_full[...,None,None] * z1
    z_tilde_self = z_tilde_full[B_emp:]                # (B_self,T,Sz,Dz)

    # --- Ramp weights w(sigma) ---
    w_emp  = 0.9 * sigma_emp  + 0.1
    w_self = 0.9 * sigma_self + 0.1

    # --- Half-step metadata for self rows (names mirror b', z', b'') ---
    d_half            = d_self / 2.0                                         # (B_self,T)
    step_idx_half     = step_idx_self + 1                                    # halve step → double K
    sigma_plus        = sigma_self + d_half                                  # σ + d/2
    sigma_idx_plus    = sigma_idx_self + (k_max * d_half).astype(jnp.int32)  # global grid shift for σ + d/2

    def loss_and_aux(p):
        dyn_vars = with_params(dynamics_vars, p)
        drop_main, drop_h1, drop_h2 = jax.random.split(drop_key, 3)

        # ---------- ONE fused main forward (emp + self) ----------
        z1_hat_full = dynamics.apply(
            dyn_vars, actions_full, step_idx_full, sigma_idx_full, z_tilde_full,
            rngs={"dropout": drop_main}, deterministic=False,
        )  # (B,T,Sz,Dz)

        # Split outputs
        z1_hat_emp  = z1_hat_full[:B_emp]     # (B_emp,T,Sz,Dz)
        z1_hat_self = z1_hat_full[B_emp:]     # (B_self,T,Sz,Dz)

        # ---------- Empirical flow loss ----------
        flow_per = jnp.mean((z1_hat_emp - z1[:B_emp])**2, axis=(2,3))        # (B_emp,T)
        loss_emp = jnp.mean(flow_per * w_emp)

        # ---------- Self-consistency (bootstrap) ----------
        # b' = (f(z_tilde, sigma, d/2) - z_tilde) / (1 - sigma)
        z1_hat_half1 = dynamics.apply(
            dyn_vars, actions_full[B_emp:], step_idx_half, sigma_idx_self, z_tilde_self,
            rngs={"dropout": drop_h1}, deterministic=False,
        )
        b_prime = (z1_hat_half1 - z_tilde_self) / (1.0 - sigma_self)[...,None,None]

        # z' = z_tilde + b' * (d/2)
        z_prime = z_tilde_self + b_prime * d_half[...,None,None]

        # b'' = (f(z', sigma + d/2, d/2) - z') / (1 - (sigma + d/2))
        z1_hat_half2 = dynamics.apply(
            dyn_vars, actions_full[B_emp:], step_idx_half, sigma_idx_plus, z_prime,
            rngs={"dropout": drop_h2}, deterministic=False,
        )
        b_doubleprime = (z1_hat_half2 - z_prime) / (1.0 - sigma_plus)[...,None,None]

        # vhat_sigma = (z1_hat_self - z_tilde_self) / (1 - sigma)
        # vbar_target = stopgrad((b' + b'') / 2)
        vhat_sigma = (z1_hat_self - z_tilde_self) / (1.0 - sigma_self)[...,None,None]
        vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)

        # L_boot in x-space: (1 - sigma)^2 * || vhat_sigma - vbar_target ||^2
        boot_per = (1.0 - sigma_self)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))  # (B_self,T)
        loss_self = jnp.mean(boot_per * w_self)

        # ---------- Combine (row-weighted) ----------
        loss = ((loss_emp * B_emp) + (loss_self * B_self)) / B

        aux = {
            "loss": loss,
            "flow_loss": jnp.mean(flow_per),
            "bootstrap_loss": jnp.mean(boot_per),
            "weighted_flow_loss": loss_emp,
            "weighted_bootstrap_loss": loss_self,
        }
        return loss, aux

    (loss_val, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(params)
    updates, opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    new_vars = with_params(dynamics_vars, new_params)
    return new_params, opt_state, new_vars, aux

def _read_only_tokenizer_meta(tokenizer_ckpt_dir: str):
    """Restore just the JSON meta from a tokenizer checkpoint directory."""
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    return latest, restored.meta


def load_pretrained_encoder_params(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    enc_vars,
    sample_patches_btnd,
):
    """Return enc_vars with restored encoder params from tokenizer ckpt (discard decoder)."""
    # -- (A) read meta only with a meta-only manager
    latest, meta = _read_only_tokenizer_meta(tokenizer_ckpt_dir)
    if "enc_kwargs" not in meta or "dec_kwargs" not in meta:
        raise ValueError("Tokenizer checkpoint meta missing enc_kwargs/dec_kwargs")
    enc_kwargs = meta["enc_kwargs"]
    dec_kwargs = meta["dec_kwargs"]

    # -- (B) build abstract trees that match the saved structure
    dec = Decoder(**dec_kwargs)
    rng_e1, rng_e2, rng_d1 = jax.random.split(rng, 3)

    B, T = sample_patches_btnd.shape[:2]
    n_lat = enc_kwargs["n_latents"]
    d_b   = enc_kwargs["d_bottleneck"]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)

    dec_vars = dec.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)

    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)

    state_example = make_state(
        params=packed_example,
        opt_state=opt_state_example,
        rng=rng_e2,
        step=0,
    )
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    # -- (C) now do a full restore (state + meta) using a manager with both items
    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        )
    )

    packed_params = restored.state["params"]        # {"enc": ..., "dec": ...}
    enc_params = packed_params["enc"]
    new_enc_vars = with_params(enc_vars, enc_params)
    return new_enc_vars, meta

def make_dynamics_meta(
    *,
    enc_kwargs: dict,
    dynamics_kwargs: dict,
    H: int, W: int, C: int,
    patch: int,
    k_max: int,
    packing_factor: int,
    n_s: int,
    tokenizer_ckpt_dir: str | None = None,
):
    return {
        "enc_kwargs": enc_kwargs,
        "dynamics_kwargs": dynamics_kwargs,
        "H": H, "W": W, "C": C, "patch": patch,
        "k_max": k_max,
        "packing_factor": packing_factor,
        "n_s": n_s,
        "tokenizer_ckpt_dir": tokenizer_ckpt_dir,
    }

if __name__ == "__main__":
    log_dir = Path("./logs"); log_dir.mkdir(parents=True, exist_ok=True)
    run_name = "test_dynamics"
    run_dir = log_dir / run_name; run_dir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(0)
    # dataset parameters ...
    B, T, H, W, C = 64, 64, 32, 32, 3
    B_self = int(0.25 * B)
    pixels_per_step = 2 # how many pixels the agent moves per step
    size_min = 6 # minimum size of the square
    size_max = 14 # maximum size of the square
    hold_min = 4 # how long the agent holds a direction for
    hold_max = 9 # how long the agent holds a direction for

    patch = 4
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = 256

    next_batch = make_iterator(B, T, H, W, C, pixels_per_step, size_min, size_max, hold_min, hold_max)
    rng, batch_rng = jax.random.split(rng)
    rng, (frames, actions) = next_batch(rng)

    # ----- models -----
    enc_n_latents, enc_d_bottleneck = 16, 32
    enc_kwargs = {
        "d_model": 64, "n_latents": enc_n_latents, "n_patches": num_patches,
        "n_heads": 4, "depth": 8, "dropout": 0.05,
        "d_bottleneck": enc_d_bottleneck, "mae_p_min": 0.0, "mae_p_max": 0.0, # disable MAE for inference.
         "time_every": 4,
    }
    packing_factor = 2
    n_s = enc_n_latents // packing_factor
    dynamics_kwargs = {
        "d_model": 128, "n_s": n_s, "d_spatial": enc_d_bottleneck * packing_factor,
        "d_bottleneck": enc_d_bottleneck, "k_max": k_max, "n_r": 10,
        "n_heads": 4, "depth": 8, "dropout": 0.0
    }

    encoder = Encoder(**enc_kwargs)
    dynamics = Dynamics(**dynamics_kwargs)

    init_patches = temporal_patchify(frames, patch)
    rng, enc_vars, dynamics_vars = init_models(
        rng, encoder, dynamics, init_patches, B, T, enc_n_latents, enc_d_bottleneck,
        packing_factor, n_s
    )

    # ====== (A) Optional: load a pretrained encoder from tokenizer checkpoints ======
    # Set this path if you want to load a trained encoder; otherwise leave as None to use fresh init.
    TOKENIZER_CKPT_DIR = "/home/edward/projects/tiny_dreamer_4/logs/test/checkpoints" 
    if TOKENIZER_CKPT_DIR is not None:
        # use init_patches as sample to build shapes if needed
        enc_vars, tok_meta = load_pretrained_encoder_params(
            TOKENIZER_CKPT_DIR,
            rng=rng,
            encoder=encoder,
            enc_vars=enc_vars,
            sample_patches_btnd=init_patches,
        )
        print(f"[encoder] Restored pretrained encoder params from: {TOKENIZER_CKPT_DIR}")
    # ----- dynamics trainables -----
    params = dynamics_vars["params"]
    tx = optax.adamw(1e-4)
    opt_state = tx.init(params)
    max_steps = 1_000_000_000

    # ====== (B) Orbax manager for dynamics run + try to restore ======
    ckpt_dir = (run_dir / "checkpoints")
    mngr = make_manager(ckpt_dir, max_to_keep=5, save_interval_steps=10_000)

    meta = make_dynamics_meta(
        enc_kwargs=enc_kwargs,
        dynamics_kwargs=dynamics_kwargs,
        H=H, W=W, C=C, patch=patch,
        k_max=k_max, packing_factor=packing_factor, n_s=n_s,
        tokenizer_ckpt_dir=TOKENIZER_CKPT_DIR
    )

    # Build example trees for safe restore
    state_example = make_state(params, opt_state, rng, step=0)
    restored = try_restore(mngr, state_example, meta)

    start_step = 0
    if restored is not None:
        latest_step, r = restored
        params     = r.state["params"]
        opt_state  = r.state["opt_state"]
        rng        = r.state["rng"]
        start_step = int(r.state["step"])
        dynamics_vars = with_params(dynamics_vars, params)
        print(f"[dynamics] Restored checkpoint step={latest_step} from {ckpt_dir}")

    # ====== (C) Training loop with periodic save ======
    # data_rng, rng = jax.random.split(rng)
    # fix data_rng for now for debugging purposes
    data_rng = jax.random.PRNGKey(0)
    try:
        for step in range(start_step, max_steps):
            data_start_t = time()
            # rng, (frames, actions) = next_batch(rng)
            # fix data_rng for now for debugging purposes
            _, (frames, actions) = next_batch(data_rng)
            data_t = time() - data_start_t

            train_start_t = time()
            rng, master_key = jax.random.split(rng)
            params, opt_state, dynamics_vars, aux = train_step_efficient(
                encoder, dynamics, tx, params, opt_state, enc_vars, dynamics_vars,
                frames, actions,
                patch=patch, B=B, T=T, B_self=B_self,
                master_key=master_key, step=step,
                packing_factor=packing_factor, n_s=n_s, k_max=k_max,
            )
            train_t = time() - train_start_t

            # logging …
            if step % 100 == 0:
                """ 
                aux = {
                    "loss": loss,
                    "flow_loss": jnp.mean(flow_per),
                    "bootstrap_loss": jnp.mean(boot_per),
                    "weighted_flow_loss": loss_emp,
                    "weighted_bootstrap_loss": loss_self,
                }"""
                print(f"step {step:05d} | flow_loss={float(aux['flow_loss']):.6f} | b_loss={float(aux['bootstrap_loss']):.6f} | weighted_flow_loss={float(aux['weighted_flow_loss']):.6f} | weighted_b_loss={float(aux['weighted_bootstrap_loss']):.6f} | time={train_t:.3f}s")

            # save (async) when policy says we should
            state = make_state(params, opt_state, rng, step)
            maybe_save(mngr, step, state, meta)

    finally:
        mngr.wait_until_finished()
