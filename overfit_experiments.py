# overfit_experiments.py
# A0→A5 ladder for single-video overfit diagnostics with toggles for k_max, σ sampling, multi-step,
# bootstrap, and pixel loss. Saves logs + GT|FLOOR|PRED triptychs per rung.

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from time import time
from typing import Sequence, Optional

import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import imageio.v2 as imageio
from functools import partial

from models import Encoder, Decoder, Dynamics
from data import make_iterator
from utils import (
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
    make_state, make_manager, with_params, pack_mae_params,
)

from sampler_unified import SamplerConfig, sample_video

# ==============================
# Small helpers
# ==============================
def psnr_from_mse(mse):
    eps = 1e-12
    return 10.0 * jnp.log10(1.0 / jnp.maximum(mse, eps))

def _to_uint8(img_f32):
    return np.asarray(np.clip(img_f32 * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    return np.concatenate(imgs_hwC, axis=1)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_py(obj):
    """Convert JAX/NumPy scalars/arrays in a nested structure to JSON-friendly Python types."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if hasattr(obj, "item") and callable(getattr(obj, "item", None)):
        try:
            return _to_py(obj.item())
        except Exception:
            pass
    if isinstance(obj, (list, tuple)):
        return [_to_py(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_py(v) for k, v in obj.items()}
    return obj

# --- tokenizer ckpt restore (your orbax layout) ---
def load_pretrained_tokenizer(
    tokenizer_ckpt_dir: str,
    *,
    rng: jnp.ndarray,
    encoder: Encoder,
    decoder: Decoder,
    enc_vars,
    dec_vars,
    sample_patches_btnd,
):
    meta_mngr = make_manager(tokenizer_ckpt_dir, item_names=("meta",))
    latest = meta_mngr.latest_step()
    if latest is None:
        raise FileNotFoundError(f"No tokenizer checkpoint found in {tokenizer_ckpt_dir}")
    restored_meta = meta_mngr.restore(latest, args=ocp.args.Composite(meta=ocp.args.JsonRestore()))
    meta = restored_meta.meta
    enc_kwargs = meta["enc_kwargs"]
    n_lat, d_b = enc_kwargs["n_latents"], enc_kwargs["d_bottleneck"]

    # build dummy trees for state restore
    rng_e1, rng_d1 = jax.random.split(rng)
    B, T = sample_patches_btnd.shape[:2]
    fake_z = jnp.zeros((B, T, n_lat, d_b), dtype=jnp.float32)
    dec_vars = decoder.init({"params": rng_d1, "dropout": rng_d1}, fake_z, deterministic=True)

    packed_example = pack_mae_params(enc_vars, dec_vars)
    tx_dummy = optax.adamw(1e-4)
    opt_state_example = tx_dummy.init(packed_example)
    state_example = make_state(packed_example, opt_state_example, rng_e1, step=0)
    abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state_example)

    tok_mngr = make_manager(tokenizer_ckpt_dir, item_names=("state", "meta"))
    restored = tok_mngr.restore(
        latest,
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            meta=ocp.args.JsonRestore(),
        ),
    )
    packed_params = restored.state["params"]
    enc_params = packed_params["enc"]
    dec_params = packed_params["dec"]
    new_enc_vars = with_params(enc_vars, enc_params)
    new_dec_vars = with_params(dec_vars, dec_params)
    print(f"[tokenizer] Restored encoder/decoder from {tokenizer_ckpt_dir} (step {latest})")
    return new_enc_vars, new_dec_vars, meta


# ==============================
# Config & rungs
# ==============================
@dataclass(frozen=True)
class ExpConfig:
    # data
    B: int = 8
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2
    size_min: int = 10
    size_max: int = 10
    hold_min: int = 6
    hold_max: int = 6
    diversify_data: bool = False

    # encoder/decoder/dynamics sizes
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 4
    n_heads: int = 4
    packing_factor: int = 2
    n_r: int = 2  # registers; set 0–2 for A0

    # schedule & losses
    k_max: int = 4                 # start tiny, grow in later rungs
    sigma_sampling: bool = False   # A1 turns this on
    multi_step_bins: bool = False  # A2 turns this on
    use_bootstrap: bool = False    # A3 turns this on
    use_pixel_loss: bool = False   # A4 turns this on
    lambda_pix: float = 0.05

    # train
    max_steps: int = 2_000
    log_every: int = 200
    lr: float = 3e-4

    # IO
    run_name: str = "overfit_rungs"
    rung_id: str = "A0"
    tokenizer_ckpt: str = "/home/edward/projects/tiny_dreamer_4/logs/test/checkpoints"


def rung_A0_easy() -> ExpConfig:
    # 1 video, fixed bg/fg/size, k_max=4, fixed sigma, single step bin, no bootstrap, latent-only loss
    return ExpConfig(
        k_max=4,
        sigma_sampling=False,
        multi_step_bins=False,
        use_bootstrap=False,
        use_pixel_loss=False,
        n_r=0,
        dyn_depth=4,
        rung_id="A0_easy_minimal",
        max_steps=3000,
    )

def rung_A1_sigma() -> ExpConfig:
    # add sigma sampling on the emax grid
    return ExpConfig(
        k_max=4,
        sigma_sampling=True,
        multi_step_bins=False,
        use_bootstrap=False,
        use_pixel_loss=False,
        rung_id="A1_sigma_sampling",
        max_steps=4000,
    )

def rung_A2_multibin() -> ExpConfig:
    # allow two step bins (emax-1 and emax)
    return ExpConfig(
        k_max=8,
        sigma_sampling=True,
        multi_step_bins=True,
        use_bootstrap=False,
        use_pixel_loss=False,
        rung_id="A2_multi_step_bins",
        max_steps=6000,
    )

def rung_A3_bootstrap() -> ExpConfig:
    # enable bootstrap (with small k_max to keep it easy)
    return ExpConfig(
        k_max=8,
        sigma_sampling=True,
        multi_step_bins=True,
        use_bootstrap=True,
        use_pixel_loss=False,
        rung_id="A3_bootstrap",
        max_steps=8000,
    )

def rung_A4_pixel() -> ExpConfig:
    # add pixel loss to reduce background drift
    return ExpConfig(
        k_max=8,
        sigma_sampling=True,
        multi_step_bins=True,
        use_bootstrap=True,
        use_pixel_loss=True,
        lambda_pix=0.05,
        rung_id="A4_pixel_aux",
        max_steps=8000,
    )

def rung_A5_diverse() -> ExpConfig:
    # increase data difficulty: unlock size/color diversity, keep modest schedule
    return ExpConfig(
        k_max=16,
        sigma_sampling=True,
        multi_step_bins=True,
        use_bootstrap=True,
        use_pixel_loss=False,
        diversify_data=True,
        lambda_pix=0.05,
        rung_id="A5_diverse_data",
        max_steps=10_000,
        n_r=4,
        dyn_depth=6,
    )


# ==============================
# Schedule utilities (jit-friendly)
# ==============================
@jax.jit
def _fixed_indices(z1_btSd, k_max: int):
    B, T = z1_btSd.shape[:2]
    emax = jnp.log2(jnp.asarray(k_max)).astype(jnp.int32)
    step_idx = jnp.full((B, T), emax, dtype=jnp.int32)
    sigma_val = 1.0 - (1.0 / k_max)
    sigma = jnp.full((B, T), sigma_val, jnp.float32)
    sigma_idx = (sigma * k_max).astype(jnp.int32)
    return step_idx, sigma, sigma_idx

@jax.jit
def _sigma_sample_for_step(rng, k_max: int, step_idx_bt: jnp.ndarray):
    """
    Sample σ on the valid grid for the given per-element step_idx.
    Shapes are taken from step_idx_bt.shape to avoid tracer shapes.
    """
    K = (1 << step_idx_bt)  # (B,T)
    # Use the concrete (compiled) shape carried by step_idx_bt
    u = jax.random.uniform(rng, step_idx_bt.shape, dtype=jnp.float32)  # (B,T)
    j_idx = jnp.floor(u * K.astype(jnp.float32)).astype(jnp.int32)     # 0..K-1
    sigma = j_idx.astype(jnp.float32) / K.astype(jnp.float32)          # (B,T)
    sigma_idx = j_idx * (k_max // K)                                    # (B,T)
    return sigma, sigma_idx

@jax.jit
def _choose_step_bins(rng, k_max: int, template_bt: jnp.ndarray):
    """
    Pick per-element step_idx from {emax-1, emax}, using template_bt only for shape.
    """
    emax = jnp.log2(jnp.asarray(k_max)).astype(jnp.int32)
    step_lo = jnp.maximum(emax - 1, 0)
    step_hi = emax
    # Random mask with the same (B,T) shape:
    coin = jax.random.bernoulli(rng, 0.5, template_bt.shape)
    step_idx = jnp.where(coin, step_lo, step_hi).astype(jnp.int32)
    return step_idx



# ==============================
# Train steps (two variants)
# ==============================
def make_train_step(static_flags):
    """
    Builds a jitted train step with the right static flags:
      static_flags = dict(
          sigma_sampling: bool,
          multi_step_bins: bool,
          use_bootstrap: bool,
          use_pixel_loss: bool,
          patch: int,
          n_s: int,
          k_max: int,
          packing_factor: int,
      )
    """
    @partial(
        jax.jit,
        static_argnames=(
            "encoder","decoder","dynamics","tx",
            "patch","n_s","k_max","packing_factor",
            "sigma_sampling","multi_step_bins","use_bootstrap","use_pixel_loss",
        ),
    )
    def train_step(
        encoder, decoder, dynamics, tx,
        params, opt_state,
        enc_vars, dec_vars, dyn_vars,
        frames, actions,
        *,
        patch, n_s, k_max, packing_factor,
        sigma_sampling: bool,
        multi_step_bins: bool,
        use_bootstrap: bool,
        use_pixel_loss: bool,
        lambda_pix: float,
        master_key: jnp.ndarray,
    ):
        patches_btnd = temporal_patchify(frames, patch)

        # split RNGs deterministically
        key_enc, key_noise, key_sigma, key_step, key_drop = jax.random.split(master_key, 5)

        # Encoder (frozen) — pass MAE RNG
        z_btLd, _ = encoder.apply(
            enc_vars, patches_btnd,
            rngs={"mae": key_enc},
            deterministic=True
        )
        z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=packing_factor)

        # Step / sigma
        if multi_step_bins:
            step_idx_full = _choose_step_bins(key_step, k_max, z1[:, :, 0, 0])
        else:
            step_idx_full, _, _ = _fixed_indices(z1, k_max)

        if sigma_sampling:
            sigma_full, sigma_idx_full = _sigma_sample_for_step(key_sigma, k_max, step_idx_full)
        else:
            _, sigma_full, sigma_idx_full = _fixed_indices(z1, k_max)

        # Corrupt: z_tilde = (1-σ) z0 + σ z1
        z0 = jax.random.normal(key_noise, z1.shape, dtype=z1.dtype)
        z_tilde = (1.0 - sigma_full)[...,None,None] * z0 + sigma_full[...,None,None] * z1

        def loss_and_metrics(p):
            local_dyn = {"params": p, **{k:v for k,v in dyn_vars.items() if k != "params"}}

            # Main forward (full step)
            z1_hat = dynamics.apply(
                local_dyn, actions, step_idx_full, sigma_idx_full, z_tilde,
                rngs={"dropout": key_drop},
                deterministic=False
            )

            flow_per = jnp.mean((z1_hat - z1) ** 2, axis=(2,3))
            flow_mse = jnp.mean(flow_per)
            loss = flow_mse
            aux = {"flow_mse": flow_mse}

            # Optional bootstrap (two half-steps) — same as your earlier formula
            if use_bootstrap:
                # d = 1/2^{e}, so half-step doubles K (e+1)
                d = 1.0 / (1 << step_idx_full).astype(jnp.float32)
                d_half = d / 2.0
                step_idx_half = jnp.clip(step_idx_full + 1, 0, jnp.log2(k_max).astype(jnp.int32))
                # σ+ and indices
                sigma_plus = jnp.clip(sigma_full + d_half, 0.0, 1.0 - (1.0 / k_max))
                sigma_idx_self = sigma_idx_full
                sigma_idx_plus = jnp.floor(sigma_plus * k_max).astype(jnp.int32)

                # b' from half step 1
                z1_hat_h1 = dynamics.apply(local_dyn, actions, step_idx_half, sigma_idx_self, z_tilde, deterministic=False)
                b_prime = (z1_hat_h1 - z_tilde) / jnp.maximum(1.0 - sigma_full, 1e-6)[...,None,None]

                # z' and b''
                z_prime = z_tilde + b_prime * d_half[...,None,None]
                z1_hat_h2 = dynamics.apply(local_dyn, actions, step_idx_half, sigma_idx_plus, z_prime, deterministic=False)
                b_doubleprime = (z1_hat_h2 - z_prime) / jnp.maximum(1.0 - sigma_plus, 1e-6)[...,None,None]

                vhat_sigma = (z1_hat - z_tilde) / jnp.maximum(1.0 - sigma_full, 1e-6)[...,None,None]
                vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)

                boot_per = (1.0 - sigma_full)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))
                boot_mse = jnp.mean(boot_per)
                aux["bootstrap_mse"] = boot_mse
                loss = loss + boot_mse

            # Optional pixel loss
            if use_pixel_loss:
                z1_hat_btLd = unpack_spatial_to_bottleneck(z1_hat, n_s=n_s, k=packing_factor)
                dec_pred = decoder.apply(dec_vars, z1_hat_btLd, deterministic=True)
                dec_gt = patches_btnd
                pix_mse = jnp.mean((dec_pred - dec_gt) ** 2)
                aux["pix_mse"] = pix_mse
                loss = loss + (lambda_pix * pix_mse)

            return loss, aux

        (loss_val, aux), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, aux

    return train_step


# ==============================
# Running one experiment
# ==============================
def _sampler_regimes_for_rung(cfg: ExpConfig, *, ctx_length: int):
    """
    Valid regimes:
      - A0: TF only, start fixed(1-1/K), finest.
      - A1: Finest; TF starts {pure, fixed(0.5), random}; AR {pure only}.
      - A2: Finest + shortcut(2/K); TF starts {pure, fixed(0.5), random};
            AR {pure only, finest only}  (no shortcut-AR without bootstrap).
      - A3: Finest + shortcut(2/K,4/K); TF starts {pure, fixed(0.5), random};
            AR {pure for finest and shortcut}.
      - A4: Same step set as A3; TF starts {pure, random}; AR {pure}.
      - A5: Same as A4; AR horizon capped to 16.
    """
    from sampler_unified import SamplerConfig

    def d_list_for(cfg: ExpConfig):
        ds = []
        if cfg.rung_id.startswith("A2"):
            ds = [2.0 / cfg.k_max]
        elif cfg.rung_id.startswith(("A3", "A4", "A5")):
            ds = [2.0 / cfg.k_max, 4.0 / cfg.k_max]
        return [d for d in ds if d <= 1.0]

    # horizon per rung
    base_hz = cfg.T - ctx_length
    horizon = min(base_hz, 16) if cfg.rung_id.startswith("A5") else base_hz
    assert horizon > 0, "ctx_length must be < T"

    common = dict(
        k_max=cfg.k_max,
        horizon=horizon,
        ctx_length=ctx_length,
        add_ctx_noise_std=0.0,
        H=cfg.H, W=cfg.W, C=cfg.C, patch=cfg.patch,
        n_s=cfg.enc_n_latents // cfg.packing_factor,
        packing_factor=cfg.packing_factor,
    )

    regimes: list[tuple[str, SamplerConfig]] = []

    # ---- A0: TF-only, near-clean, finest
    if cfg.rung_id.startswith("A0"):
        regimes.append((
            "finest_fixed_near_TF",
            SamplerConfig(schedule="finest", rollout="teacher_forced",
                          start_mode="fixed", tau0_fixed=1.0 - 1.0/cfg.k_max,
                          **common)
        ))
        return regimes

    # ---- TF start sets by rung
    if cfg.rung_id.startswith(("A4", "A5")):
        tf_starts = [("pure", dict(start_mode="pure")),
                     ("random", dict(start_mode="random"))]
    elif cfg.rung_id.startswith(("A1", "A2", "A3")):
        tf_starts = [("pure", dict(start_mode="pure")),
                     ("fixed05", dict(start_mode="fixed", tau0_fixed=0.5)),
                     ("random", dict(start_mode="random"))]
    else:
        tf_starts = [("pure", dict(start_mode="pure"))]

    # ---- Finest regimes
    # TF: all allowed starts
    for s_name, s_kwargs in tf_starts:
        tag = f"finest_{s_name}_TF"
        regimes.append((
            tag, SamplerConfig(schedule="finest", rollout="teacher_forced",
                               **s_kwargs, **common)
        ))

    # AR: pure only
    regimes.append((
        "finest_pure_AR",
        SamplerConfig(schedule="finest", rollout="autoregressive",
                      start_mode="pure", **common)
    ))

    # ---- Shortcut regimes (A2+)
    for d in d_list_for(cfg):
        K = int(round(1.0/d))

        # TF: all allowed starts
        for s_name, s_kwargs in tf_starts:
            tag = f"shortcut_d{K}_{s_name}_TF"
            regimes.append((
                tag, SamplerConfig(schedule="shortcut", d=d, rollout="teacher_forced",
                                   **s_kwargs, **common)
            ))

        # AR: pure only, but only if trained with bootstrap
        if cfg.use_bootstrap:
            tag = f"shortcut_d{K}_pure_AR"
            regimes.append((
                tag, SamplerConfig(schedule="shortcut", d=d, rollout="autoregressive",
                                   start_mode="pure", **common)
            ))

    return regimes


def run_one(config: ExpConfig):
    # IO
    out_dir = _ensure_dir(Path("./overfit_exps") / config.run_name / config.rung_id)

    # Data (optionally diversified)
    B, T, H, W, C = config.B, config.T, config.H, config.W, config.C
    if config.diversify_data:
        B = 8
        size_min, size_max = 6, 14
        hold_min, hold_max = 4, 9
        fg_min_color, fg_max_color = 0, 255
        bg_min_color, bg_max_color = 0, 255
    else:
        size_min = config.size_min
        size_max = config.size_max
        hold_min = config.hold_min
        hold_max = config.hold_max
        fg_min_color, fg_max_color = 128, 128
        bg_min_color, bg_max_color = 255, 255

    next_batch = make_iterator(
        B, T, H, W, C,
        pixels_per_step=config.pixels_per_step,
        size_min=size_min, size_max=size_max,
        hold_min=hold_min, hold_max=hold_max,
        fg_min_color=fg_min_color, fg_max_color=fg_max_color,
        bg_min_color=bg_min_color, bg_max_color=bg_max_color,
    )
    rng = jax.random.PRNGKey(0)
    _, (frames, actions) = next_batch(rng)

    # Models
    patch = config.patch
    num_patches = (H // patch) * (W // patch)
    D_patch = patch * patch * C
    k_max = config.k_max

    enc_kwargs = dict(
        d_model=config.d_model_enc,
        n_latents=config.enc_n_latents,
        n_patches=num_patches,
        n_heads=config.n_heads,
        depth=config.enc_depth,
        dropout=0.0,
        d_bottleneck=config.enc_d_bottleneck,
        mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=config.d_model_enc,
        n_heads=config.n_heads,
        depth=config.dec_depth,
        n_latents=config.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    n_s = config.enc_n_latents // config.packing_factor
    dyn_kwargs = dict(
        d_model=config.d_model_dyn,
        d_bottleneck=config.enc_d_bottleneck,
        d_spatial=config.enc_d_bottleneck * config.packing_factor,
        n_s=n_s, n_r=config.n_r,
        n_heads=config.n_heads, depth=config.dyn_depth,
        dropout=0.0, k_max=k_max,
        time_every=4, latents_only_time=False,
    )

    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)

    patches_btnd = temporal_patchify(frames, patch)
    rng = jax.random.PRNGKey(0)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((B, T, config.enc_n_latents, config.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # Load tokenizer params
    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        config.tokenizer_ckpt, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    # Fixed MAE RNG for evaluation forwards (decoding floor/preds)
    mae_eval_key = jax.random.PRNGKey(777)

    # Build initial z1 to shape the dynamics init
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=config.packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((B, T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((B, T), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions, step_idx, sigma_idx, z1)

    params = dyn_vars["params"]
    tx = optax.adam(config.lr)
    opt_state = tx.init(params)

    # Baseline decoder floor (same encode key)
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    dec_floor = decoder.apply(dec_vars, z_btLd, deterministic=True)
    floor_mse = jnp.mean((dec_floor - patches_btnd) ** 2)
    print(f"[{config.rung_id}] floor_mse={float(floor_mse):.6g} (PSNR={float(psnr_from_mse(floor_mse)):.2f} dB)")

    # Build train_step with static flags
    train_step = make_train_step(dict(
        sigma_sampling=config.sigma_sampling,
        multi_step_bins=config.multi_step_bins,
        use_bootstrap=config.use_bootstrap,
        use_pixel_loss=config.use_pixel_loss,
        patch=config.patch, n_s=n_s, k_max=k_max, packing_factor=config.packing_factor,
    ))

    # Train
    start_t = time()
    for step in range(config.max_steps):
        rng, key = jax.random.split(rng)
        params, opt_state, aux = train_step(
            encoder, decoder, dynamics, tx,
            params, opt_state, enc_vars, dec_vars, dyn_vars,
            frames, actions,
            patch=config.patch, n_s=n_s, k_max=k_max, packing_factor=config.packing_factor,
            sigma_sampling=config.sigma_sampling,
            multi_step_bins=config.multi_step_bins,
            use_bootstrap=config.use_bootstrap,
            use_pixel_loss=config.use_pixel_loss,
            lambda_pix=config.lambda_pix,
            master_key=key,
        )
        if step % config.log_every == 0:
            msg = [f"[{config.rung_id}] step {step:06d}"]
            msg.append(f"flow_mse={float(aux['flow_mse']):.6g}")
            if config.use_bootstrap and "bootstrap_mse" in aux:
                msg.append(f"boot_mse={float(aux['bootstrap_mse']):.6g}")
            if config.use_pixel_loss and "pix_mse" in aux:
                pm = float(aux["pix_mse"]); ps = float(psnr_from_mse(aux["pix_mse"]))
                msg.append(f"pix_mse={pm:.6g} (PSNR={ps:.2f} dB)")
            msg.append(f"floor_mse={float(floor_mse):.6g}")
            msg.append(f"t={time()-start_t:.1f}s")
            print(" | ".join(msg))

    # Final decode (eval path with fixed MAE key)
    # ========= Unified sampler evaluations (per regime) =========
    dyn_vars_final = with_params(dyn_vars, params)
    ctx_length = min(32, config.T - 1)  # same context you’ve been using; keep < T
    eval_regimes = _sampler_regimes_for_rung(config, ctx_length=ctx_length)

    for tag, sconf in eval_regimes:
        # Collect sampler logs (plan + per-frame "step" dicts)
        logs: list[dict] = []

        def _hook(d):
            logs.append(_to_py(d))

        # plumb deterministic keys + debug
        sconf.mae_eval_key = mae_eval_key
        sconf.rng_key = rng
        sconf.debug = True                # enable sampler logging
        sconf.debug_hook = _hook          # capture plan + per-step summaries

        start_t_samp = time()
        pred_frames, floor_frames, gt_frames = sample_video(
            encoder=encoder, decoder=decoder, dynamics=dynamics,
            enc_vars=enc_vars, dec_vars=dec_vars, dyn_vars=dyn_vars_final,
            frames=frames, actions=actions, config=sconf,
        )
        samp_dt = time() - start_t_samp

        # --- write debug logs next to the media ---
        plan = next((x for x in logs if x.get("kind") == "plan"), None)
        steps = [x for x in logs if x.get("kind") == "step"]

        plan_path  = out_dir / f"{tag}_plan.json"
        steps_path = out_dir / f"{tag}_steps.json"

        try:
            if plan is not None:
                plan["elapsed_sec"] = float(samp_dt)
                with open(plan_path, "w") as f:
                    json.dump(plan, f, indent=2)
            if steps:
                with open(steps_path, "w") as f:
                    json.dump(steps, f, indent=2)
            print(f"[{config.rung_id}/{tag}] wrote {plan_path.name} and {steps_path.name}")
        except Exception as e:
            print(f"[{config.rung_id}/{tag}] JSON log write skipped ({e})")

        # Metrics over the predicted horizon only
        HZ = sconf.horizon
        pred_eval = pred_frames[:, -HZ:]   # (B, HZ, H, W, C)
        gt_eval   = gt_frames[:,  -HZ:]
        mse = jnp.mean((pred_eval - gt_eval) ** 2)
        psnr = psnr_from_mse(mse)
        print(f"[{config.rung_id}/{tag}] eval_horizon_mse={float(mse):.6g} (PSNR={float(psnr):.2f} dB) | time={samp_dt:.2f}s")

        # === Visualization: GT | FLOOR | PRED for (context + horizon) on batch[0]
        gt_np    = _to_uint8(gt_frames[0])      # (ctx+HZ, H, W, C)
        floor_np = _to_uint8(floor_frames[0])   # (ctx+HZ, H, W, C)
        pred_np  = _to_uint8(pred_frames[0])    # (ctx+HZ, H, W, C)

        trip_frames = [ _stack_wide(gt_np[t], floor_np[t], pred_np[t]) for t in range(gt_np.shape[0]) ]

        gif_path = out_dir / f"{tag}_gt_floor_pred.gif"
        imageio.mimsave(gif_path, trip_frames, duration=1/25)
        print(f"[{config.rung_id}/{tag}] wrote {gif_path.resolve()}")

        mp4_path = out_dir / f"{tag}_gt_floor_pred.mp4"
        try:
            with imageio.get_writer(mp4_path, fps=25, codec="libx264", quality=8) as w:
                for fr in trip_frames:
                    w.append_data(fr)
            print(f"[{config.rung_id}/{tag}] wrote {mp4_path.resolve()}")
        except Exception as e:
            print(f"[{config.rung_id}/{tag}] MP4 write skipped ({e}); GIF saved.")

    # Save config used
    (out_dir / "config.txt").write_text("\n".join([f"{k}={v}" for k, v in asdict(config).items()]))

    # return float(floor_mse), float(final_pix_mse)


# ==============================
# Main: run A0→A5
# ==============================
if __name__ == "__main__":
    rungs: Sequence[ExpConfig] = [
        rung_A0_easy(),
        rung_A1_sigma(),
        rung_A2_multibin(),
        rung_A3_bootstrap(),
        rung_A4_pixel(),
        rung_A5_diverse(),
    ]

    print("Will run rungs:")
    for r in rungs:
        print(f"  - {r.rung_id} :: k_max={r.k_max}, sigma_sampling={r.sigma_sampling}, multi_step_bins={r.multi_step_bins}, "
              f"bootstrap={r.use_bootstrap}, pixel={r.use_pixel_loss}, diverse={r.diversify_data}")

    for cfg in rungs:
        run_one(cfg)
