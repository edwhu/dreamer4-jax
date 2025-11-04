# sampler_unified.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Tuple, Optional, Dict, Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

from models import Encoder, Decoder, Dynamics
from utils import (
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
)

# ---------------------------
# Config & small utilities
# ---------------------------

StartMode   = Literal["pure", "fixed", "random"]
Schedule    = Literal["finest", "shortcut"]
RolloutMode = Literal["teacher_forced", "autoregressive"]

@dataclass
class SamplerConfig:
    k_max: int
    schedule: Schedule                      # "finest" or "shortcut"
    d: Optional[float] = None               # used iff schedule == "shortcut"
    start_mode: StartMode = "pure"          # in TF: {"pure","fixed","random"}; in AR: must be "pure"
    tau0_fixed: float = 0.5                 # used iff start_mode == "fixed"
    rollout: RolloutMode = "teacher_forced" # "teacher_forced" or "autoregressive"
    horizon: int = 1
    ctx_length: int = 32
    ctx_signal_tau: Optional[float] = None  # e.g., 0.9 for “slightly corrupt”; None or 1.0 = clean
    rng_key: Optional[jax.Array] = None
    mae_eval_key: Optional[jax.Array] = None
    # decoding sizes
    H: int = 32; W: int = 32; C: int = 3; patch: int = 4
    # tokenizer shapes
    n_s: int = 8
    packing_factor: int = 2
    # debugging (host-side only)
    debug: bool = False
    # Called with a dict. We call it twice: once with a high-level run "plan" (kind="plan"),
    # and once per step for the scalar fields (kind="step").
    debug_hook: Optional[Callable[[dict], None]] = None

def _assert_power_of_two(k: int):
    if k < 1 or (k & (k - 1)) != 0:
        raise ValueError(f"k_max must be a positive power of two, got {k}")

def _is_power_of_two_fraction(x: float) -> bool:
    if x <= 0 or x > 1: return False
    inv = round(1.0 / x)
    return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

def _step_idx_from_d(d: float, k_max: int) -> int:
    K = round(1.0 / float(d))
    if abs(1.0 / K - d) > 1e-8:
        raise ValueError(f"d={d} is not an exact 1/(power of two)")
    e = int(round(np.log2(K)))
    emax = int(round(np.log2(k_max)))
    if e > emax:
        raise ValueError(f"step bin e={e} (d={d}) is coarser than allowed emax={emax} (k_max={k_max})")
    return e

def _choose_step_size(k_max: int, schedule: Schedule, d: Optional[float]) -> float:
    _assert_power_of_two(k_max)
    if schedule == "finest":
        return 1.0 / float(k_max)
    if d is None:
        raise ValueError("schedule='shortcut' requires config.d (e.g., 1/4)")
    if not _is_power_of_two_fraction(d):
        raise ValueError(f"d must be 1/(power of two); got d={d}")
    if d < 1.0 / float(k_max):
        raise ValueError(f"d={d} is finer than d_min=1/{k_max}")
    return float(d)

def _align_to_grid(tau0: float, d: float) -> float:
    # Snap tau0 to the {0, d, 2d, ...} grid. No-op if already aligned.
    return float(np.clip(np.floor(tau0 / d) * d, 0.0, 1.0))

def _signal_idx_from_tau(tau: jnp.ndarray, k_max: int) -> jnp.ndarray:
    idx = (tau * k_max).astype(jnp.int32)
    return jnp.clip(idx, 0, k_max - 1)

def _validate_modes(cfg: SamplerConfig):
    # AR ⇒ start must be "pure"
    if cfg.rollout == "autoregressive" and cfg.start_mode != "pure":
        raise ValueError("Autoregressive rollout supports only start_mode='pure'. "
                         "Use teacher_forced for fixed/random starts.")
    # Finest vs shortcut d usage
    if cfg.schedule == "finest" and cfg.d is not None:
        raise ValueError("Provide d only for schedule='shortcut'.")
    if cfg.schedule == "shortcut" and cfg.d is None:
        raise ValueError("schedule='shortcut' requires a valid d (e.g., 1/4).")

def _tau_grid_from(k_max: int, schedule: Schedule, d_opt: Optional[float], start_tau: float) -> Tuple[jnp.ndarray, int, float, int]:
    d = _choose_step_size(k_max, schedule, d_opt)
    start_aligned = jnp.clip(jnp.floor(start_tau / d) * d, 0.0, 1.0)
    S_float = (1.0 - start_aligned) / d
    S = int(jnp.round(S_float))
    tau_seq = start_aligned + d * jnp.arange(S + 1, dtype=jnp.float32)
    tau_seq = jnp.clip(tau_seq, 0.0, 1.0)
    e = _step_idx_from_d(float(d), k_max)
    return tau_seq, S, float(d), e

# ---------- NEW: high-level plan builder (host-only) ----------

def _build_run_plan(cfg: SamplerConfig) -> dict:
    """
    Summarize the *planned* schedule before any denoising:
      - exact d, e
      - start policy and implied S (or S range)
      - tau0 grid/logical info
    """
    d_used = _choose_step_size(cfg.k_max, cfg.schedule, cfg.d)
    e = _step_idx_from_d(d_used, cfg.k_max)
    d_inv = int(round(1.0 / d_used))
    plan = {
        "kind": "plan",
        "k_max": cfg.k_max,
        "schedule": cfg.schedule,
        "d": d_used,
        "K": d_inv,
        "e": e,
        "rollout": cfg.rollout,
        "start_mode": cfg.start_mode,
        "ctx_length": cfg.ctx_length,
        "horizon": cfg.horizon,
    }
    if cfg.rollout == "autoregressive":
        # AR always starts at tau0=0
        plan["tau0_policy"] = "pure (0.0)"
        plan["S"] = int(1.0 / d_used)
        plan["S_range"] = (plan["S"], plan["S"])
        plan["tau0_grid"] = [0.0]
    else:
        # teacher-forced
        if cfg.start_mode == "pure":
            plan["tau0_policy"] = "pure (0.0)"
            plan["S"] = int(1.0 / d_used)
            plan["S_range"] = (plan["S"], plan["S"])
            plan["tau0_grid"] = [0.0]
        elif cfg.start_mode == "fixed":
            tau0a = _align_to_grid(float(np.clip(cfg.tau0_fixed, 0.0, 1.0)), d_used)
            plan["tau0_policy"] = f"fixed(aligned={tau0a:.6f})"
            plan["S"] = int(round((1.0 - tau0a) / d_used))
            plan["S_range"] = (plan["S"], plan["S"])
            # show a few grid marks near the fixed start for readability
            grid = [i * d_used for i in range(0, d_inv)]
            plan["tau0_grid"] = grid
        else:  # random
            plan["tau0_policy"] = f"random on grid {{0, d, ..., 1-d}}"
            plan["S_range"] = (1, int(1.0 / d_used))  # τ0 in {0, d, ..., 1-d}
            # Don't list all grid points if K is big; cap preview
            if d_inv <= 16:
                plan["tau0_grid"] = [i * d_used for i in range(0, d_inv)]
            else:
                plan["tau0_grid"] = ["0", "d", "...", "1-d"]
    return plan

def _emit_plan(plan: dict, hook: Optional[Callable[[dict], None]], enable_print: bool):
    if hook:
        hook(plan)
    if enable_print:
        # Compact, stable ordering
        keys = ["kind","k_max","schedule","d","K","e","rollout","start_mode",
                "ctx_length","horizon","tau0_policy","S","S_range","tau0_grid"]
        msg = {k: plan[k] for k in keys if k in plan}
        print(f"[sampler] {msg}")

# ---------------------------
# Core Single-Frame Sampler
# ---------------------------

def denoise_single_latent(
    *,
    dynamics: Dynamics,
    dyn_vars: Dict[str, Any],
    actions_ctx: jnp.ndarray,     # (B, T_ctx)
    action_curr: jnp.ndarray,     # (B, 1)
    z_ctx: jnp.ndarray,           # (B, T_ctx, N_s, D_s)
    k_max: int,
    d: float,
    start_mode: StartMode,
    tau0_fixed: float,
    rng_key: jax.Array,
    clean_target_next: Optional[jnp.ndarray],  # (B,1,N_s,D_s) if TF else None
    debug: bool = False,
    debug_hook: Optional[Callable[[dict], None]] = None,
) -> jnp.ndarray:
    B, T_ctx, N_s, D_s = z_ctx.shape
    _assert_power_of_two(k_max)
    assert actions_ctx.shape == (B, T_ctx)
    assert action_curr.shape == (B, 1)

    # 1) choose tau0
    rng_key, r_tau, r_noise = jax.random.split(rng_key, 3)
    if start_mode == "pure":
        tau0 = 0.0
    elif start_mode == "fixed":
        tau0 = float(np.clip(tau0_fixed, 0.0, 1.0))
    else:
        tau0 = float(jax.random.uniform(r_tau, (), minval=0.0, maxval=1.0))
    tau0_aligned = _align_to_grid(tau0, d) if tau0 > 0.0 else 0.0

    # 2) init corrupted latent at tau0
    z0 = jax.random.normal(r_noise, (B, 1, N_s, D_s), dtype=z_ctx.dtype)
    if tau0_aligned > 0.0 and clean_target_next is not None:
        z_t = tau0_aligned * clean_target_next + (1.0 - tau0_aligned) * z0
    else:
        z_t = z0  # AR path or TF with pure start

    # 3) tau ladder & indices
    tau_seq, S, d_used, e = _tau_grid_from(k_max, "shortcut" if d > (1.0/k_max) else "finest", d, tau0_aligned)
    tau_seq_host = list(np.asarray(tau_seq))  # host loop
    actions_full = jnp.concatenate([actions_ctx, action_curr], axis=1)  # (B, T_ctx+1)
    step_idx = jnp.full((B, T_ctx + 1), e, dtype=jnp.int32)

    if debug:
        info = {"kind":"step", "tau0": float(tau0), "tau0_aligned": float(tau0_aligned),
                "d": float(d_used), "e": int(e), "S": int(S)}
        if debug_hook: debug_hook(info)
        else: print(f"[sampler] {info}")

    # 4) iterate S steps
    for s in range(1, len(tau_seq_host)):
        tau = float(tau_seq_host[s])
        signal_idx = jnp.full((B, T_ctx + 1), _signal_idx_from_tau(jnp.asarray(tau), k_max), dtype=jnp.int32)
        z_seq = jnp.concatenate([z_ctx, z_t], axis=1)

        rng_key, drop_key = jax.random.split(rng_key)
        z_clean_pred_seq = dynamics.apply(
            dyn_vars,
            actions_full,
            step_idx,
            signal_idx,
            z_seq,
            rngs={"dropout": drop_key},
            deterministic=True,
        )
        z_clean_pred = z_clean_pred_seq[:, -1:, :, :]
        z_t = (1.0 - d_used) * z_t + d_used * z_clean_pred

    return z_t  # (B,1,N_s,D_s)

# ---------------------------
# Multi-frame rollout wrapper
# ---------------------------

def sample_video(
    *,
    encoder: Encoder,
    decoder: Decoder,
    dynamics: Dynamics,
    enc_vars: Dict[str, Any],
    dec_vars: Dict[str, Any],
    dyn_vars: Dict[str, Any],
    frames: jnp.ndarray,     # (B,T,H,W,C)
    actions: jnp.ndarray,    # (B,T)
    config: SamplerConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    B, T, H, W, C = frames.shape
    assert config.ctx_length < T, "ctx_length must be < T"
    _validate_modes(config)

    # --- One-shot plan print before any work ---
    plan = _build_run_plan(config)
    _emit_plan(plan, config.debug_hook, config.debug)

    horizon = config.horizon
    rng = config.rng_key if config.rng_key is not None else jax.random.PRNGKey(0)
    mae_key = config.mae_eval_key if config.mae_eval_key is not None else jax.random.PRNGKey(777)

    # 1) encode once (deterministic key)
    patches = temporal_patchify(frames, config.patch)
    z_btLd, _ = encoder.apply(enc_vars, patches, rngs={"mae": mae_key}, deterministic=True)
    z_all = pack_bottleneck_to_spatial(z_btLd, n_s=config.n_s, k=config.packing_factor)

    # 2) split context vs future
    z_ctx_clean = z_all[:, :config.ctx_length, :, :]
    actions_ctx = actions[:, :config.ctx_length]
    future_actions = actions[:, config.ctx_length: config.ctx_length + horizon]
    gt_future_latents = z_all[:, config.ctx_length: config.ctx_length + horizon, :, :]

    # --- Context corruption à la paper: z_ctx <- τ * z_clean + (1-τ) * N(0, I)
    if config.ctx_signal_tau is not None and config.ctx_signal_tau < 1.0:
        rng, nkey = jax.random.split(rng)
        noise = jax.random.normal(nkey, z_ctx_clean.shape, z_ctx_clean.dtype)
        tau = jnp.asarray(config.ctx_signal_tau, z_ctx_clean.dtype)
        z_ctx_clean = tau * z_ctx_clean + (1.0 - tau) * noise


    # 3) floor: decoder recon of (ctx + GT future)
    floor_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_ctx_clean, n_s=config.n_s, k=config.packing_factor),
        unpack_spatial_to_bottleneck(gt_future_latents, n_s=config.n_s, k=config.packing_factor)
    ], axis=1)
    floor_patches = decoder.apply(dec_vars, floor_btLd, deterministic=True)
    floor_frames = temporal_unpatchify(floor_patches, H, W, C, config.patch)

    # 4) choose schedule/step size
    d = _choose_step_size(config.k_max, config.schedule, config.d)

    # 5) rollout
    preds: list[jnp.ndarray] = []
    z_ctx_run = z_ctx_clean
    actions_ctx_run = actions_ctx

    for t in range(horizon):
        action_curr = future_actions[:, t:t+1]
        # TF exposes the clean target; AR does not
        z1_ref = gt_future_latents[:, t:t+1, :, :] if config.rollout == "teacher_forced" else None

        rng, step_key = jax.random.split(rng)
        z_clean_pred = denoise_single_latent(
            dynamics=dynamics,
            dyn_vars=dyn_vars,
            actions_ctx=actions_ctx_run,
            action_curr=action_curr,
            z_ctx=z_ctx_run,
            k_max=config.k_max,
            d=d,
            start_mode=config.start_mode,
            tau0_fixed=config.tau0_fixed,
            rng_key=step_key,
            clean_target_next=z1_ref,
            debug=config.debug,
            debug_hook=config.debug_hook,
        )
        preds.append(z_clean_pred)

        # advance context
        if config.rollout == "autoregressive":
            z_ctx_run = jnp.concatenate([z_ctx_run, z_clean_pred], axis=1)[:, -config.ctx_length:, :, :]
            actions_ctx_run = jnp.concatenate([actions_ctx_run, action_curr], axis=1)[:, -config.ctx_length:]
        else:  # teacher-forced
            z_ctx_run = jnp.concatenate([z_ctx_run, z1_ref], axis=1)[:, -config.ctx_length:, :, :]
            actions_ctx_run = jnp.concatenate([actions_ctx_run, action_curr], axis=1)[:, -config.ctx_length:]

    # 6) decode predictions (prepend context for viz)
    pred_latents = jnp.concatenate(preds, axis=1)
    pred_btLd = jnp.concatenate([
        unpack_spatial_to_bottleneck(z_ctx_clean, n_s=config.n_s, k=config.packing_factor),
        unpack_spatial_to_bottleneck(pred_latents, n_s=config.n_s, k=config.packing_factor),
    ], axis=1)
    pred_patches = decoder.apply(dec_vars, pred_btLd, deterministic=True)
    pred_frames = temporal_unpatchify(pred_patches, H, W, C, config.patch)

    gt_frames = frames[:, :config.ctx_length + horizon]
    return pred_frames, floor_frames, gt_frames
