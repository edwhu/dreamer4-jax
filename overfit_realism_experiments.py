# overfit_realism_experiments.py
# Streaming-batch "realism" training on synthetic data with TF training and AR evaluation.
# k_max=8 with two bins {1/8, 1/4}; σ-sampling on; multi-bin on; bootstrap after warm-up.
# Saves GT|FLOOR|PRED triptychs and a sampler plan JSON per visualization.

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict, Any
from functools import partial
import json
import time
import math

import jax
import jax.numpy as jnp
import numpy as np
import optax
import imageio.v2 as imageio
import orbax.checkpoint as ocp  # only used by tokenizer restore

from models import Encoder, Decoder, Dynamics
from data import make_iterator
from utils import (
    temporal_patchify, temporal_unpatchify,
    pack_bottleneck_to_spatial, unpack_spatial_to_bottleneck,
    make_state, make_manager, with_params, pack_mae_params,
)

from sampler_unified import SamplerConfig, sample_video

# ---------------------------
# Utilities
# ---------------------------

def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _to_uint8(img_f32):
    return np.asarray(np.clip(np.asarray(img_f32) * 255.0, 0, 255), dtype=np.uint8)

def _stack_wide(*imgs_hwC):
    return np.concatenate(imgs_hwC, axis=1)

def psnr_from_mse(mse):
    eps = 1e-12
    return 10.0 * jnp.log10(1.0 / jnp.maximum(mse, eps))

def _tile_triptychs(trip_list_hwC: list[np.ndarray], *, ncols: int = 2, pad_color: int = 0) -> np.ndarray:
    """
    Given a list of per-sample triptychs (each Hx(3W)xC), tile into a grid image.
    Returns a single H*rows x (3W)*ncols x C image.
    """
    if len(trip_list_hwC) == 0:
        raise ValueError("Empty triptych list")

    H, W3, C = trip_list_hwC[0].shape
    B = len(trip_list_hwC)
    nrows = math.ceil(B / ncols)

    # Pad with blanks if needed.
    total = nrows * ncols
    if total > B:
        blank = np.full((H, W3, C), pad_color, dtype=trip_list_hwC[0].dtype)
        trip_list_hwC = trip_list_hwC + [blank] * (total - B)

    # Build rows
    rows = []
    idx = 0
    for _ in range(nrows):
        row_imgs = trip_list_hwC[idx:idx + ncols]
        idx += ncols
        rows.append(np.concatenate(row_imgs, axis=1))  # concat across width
    grid = np.concatenate(rows, axis=0)  # stack rows vertically
    return grid  # (nrows*H, ncols*(3W), C)

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

# ---------------------------
# Training step (TF)
# ---------------------------

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
    K = (1 << step_idx_bt)  # (B,T)
    u = jax.random.uniform(rng, step_idx_bt.shape, dtype=jnp.float32)
    j_idx = jnp.floor(u * K.astype(jnp.float32)).astype(jnp.int32)   # 0..K-1
    sigma = j_idx.astype(jnp.float32) / K.astype(jnp.float32)        # in {0..(K-1)}/K
    sigma_idx = j_idx * (k_max // K)                                  # map to global bins
    return sigma, sigma_idx

@jax.jit
def _choose_step_bins(rng, k_max: int, template_bt: jnp.ndarray):
    emax = jnp.log2(jnp.asarray(k_max)).astype(jnp.int32)
    step_lo = jnp.maximum(emax - 1, 0)  # emax-1
    step_hi = emax                      # emax
    coin = jax.random.bernoulli(rng, 0.5, template_bt.shape)
    return jnp.where(coin, step_lo, step_hi).astype(jnp.int32)

def make_train_step(static_flags):
    @partial(
        jax.jit,
        static_argnames=(
            "encoder","decoder","dynamics","tx",
            "patch","n_s","k_max","packing_factor",
            "sigma_sampling","multi_step_bins","use_bootstrap",
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
        master_key: jnp.ndarray,
    ):
        patches_btnd = temporal_patchify(frames, patch)

        key_enc, key_noise, key_sigma, key_step, key_drop = jax.random.split(master_key, 5)

        # Encode (frozen)
        z_btLd, _ = encoder.apply(
            enc_vars, patches_btnd, rngs={"mae": key_enc}, deterministic=True
        )
        z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=packing_factor)  # (B,T,Ns,Ds)

        # Step / sigma bins
        if multi_step_bins:
            step_idx_full = _choose_step_bins(key_step, k_max, z1[:, :, 0, 0])
        else:
            step_idx_full, _, _ = _fixed_indices(z1, k_max)

        if sigma_sampling:
            sigma_full, sigma_idx_full = _sigma_sample_for_step(key_sigma, k_max, step_idx_full)
        else:
            _, sigma_full, sigma_idx_full = _fixed_indices(z1, k_max)

        # Corrupt
        z0 = jax.random.normal(key_noise, z1.shape, dtype=z1.dtype)
        z_tilde = (1.0 - sigma_full)[..., None, None] * z0 + sigma_full[..., None, None] * z1

        def loss_and_metrics(p):
            local_dyn = {"params": p, **{k:v for k,v in dyn_vars.items() if k != "params"}}

            # Main TF prediction
            z1_hat = dynamics.apply(
                local_dyn, actions, step_idx_full, sigma_idx_full, z_tilde,
                rngs={"dropout": key_drop}, deterministic=False
            )
            flow_per = jnp.mean((z1_hat - z1) ** 2, axis=(2,3))
            flow_mse = jnp.mean(flow_per)
            loss = flow_mse
            aux = {"flow_mse": flow_mse}

            # Optional bootstrap (two half-steps)
            if use_bootstrap:
                d = 1.0 / (1 << step_idx_full).astype(jnp.float32)  # 1/2^e
                d_half = d / 2.0
                emax = jnp.log2(k_max).astype(jnp.int32)
                step_idx_half = jnp.clip(step_idx_full + 1, 0, emax)

                sigma_plus = jnp.clip(sigma_full + d_half, 0.0, 1.0 - (1.0 / k_max))
                sigma_idx_self = sigma_idx_full
                sigma_idx_plus = jnp.floor(sigma_plus * k_max).astype(jnp.int32)

                z1_hat_h1 = dynamics.apply(local_dyn, actions, step_idx_half, sigma_idx_self, z_tilde, deterministic=False)
                b_prime = (z1_hat_h1 - z_tilde) / jnp.maximum(1.0 - sigma_full, 1e-6)[..., None, None]

                z_prime = z_tilde + b_prime * d_half[..., None, None]
                z1_hat_h2 = dynamics.apply(local_dyn, actions, step_idx_half, sigma_idx_plus, z_prime, deterministic=False)
                b_doubleprime = (z1_hat_h2 - z_prime) / jnp.maximum(1.0 - sigma_plus, 1e-6)[..., None, None]

                vhat_sigma = (z1_hat - z_tilde) / jnp.maximum(1.0 - sigma_full, 1e-6)[..., None, None]
                vbar_target = jax.lax.stop_gradient((b_prime + b_doubleprime) / 2.0)

                boot_per = (1.0 - sigma_full)**2 * jnp.mean((vhat_sigma - vbar_target)**2, axis=(2,3))
                boot_mse = jnp.mean(boot_per)
                aux["bootstrap_mse"] = boot_mse
                loss = loss + boot_mse

            return loss, aux

        (loss_val, aux), grads = jax.value_and_grad(loss_and_metrics, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, aux

    return train_step

# ---------------------------
# Eval regimes & plan JSON
# ---------------------------

def _eval_regimes_for_realism(cfg, *, ctx_length: int) -> list[tuple[str, SamplerConfig]]:
    # Always eval finest AR; also eval shortcut_d4 AR (k_max=8 -> d=1/4)
    common = dict(
        k_max=cfg.k_max,
        horizon=min(32, cfg.T - ctx_length),
        ctx_length=ctx_length,
        ctx_signal_tau=0.99,  # slight robustness noise during AR eval
        H=cfg.H, W=cfg.W, C=cfg.C, patch=cfg.patch,
        n_s=cfg.enc_n_latents // cfg.packing_factor,
        packing_factor=cfg.packing_factor,
        start_mode="pure",  # AR requires pure
        rollout="autoregressive",
    )
    regs: list[tuple[str, SamplerConfig]] = []
    regs.append(("finest_pure_AR", SamplerConfig(schedule="finest", **common)))
    regs.append(("shortcut_d4_pure_AR", SamplerConfig(schedule="shortcut", d=1/4, **common)))
    return regs

def _plan_from_sconf(s: SamplerConfig) -> Dict[str, Any]:
    # Recompute the scheduling descriptors for JSON (mirrors unified sampler logic)
    def _is_pow2_frac(x: float) -> bool:
        if x <= 0 or x > 1: return False
        inv = round(1.0 / x)
        return abs(1.0 / inv - x) < 1e-8 and (inv & (inv - 1)) == 0

    if s.schedule == "finest":
        d = 1.0 / float(s.k_max)
    else:
        if s.d is None or not _is_pow2_frac(s.d):
            raise ValueError("shortcut schedule requires d = 1/(power of two)")
        if s.d < 1.0 / float(s.k_max):
            raise ValueError("d finer than finest")
        d = float(s.d)

    tau0 = 0.0  # AR-pure
    S = int(round((1.0 - tau0) / d))
    e = int(round(np.log2(round(1.0 / d))))
    tau_seq = [round(tau0 + i*d, 6) for i in range(S + 1)]
    tau_seq[-1] = 1.0

    return dict(
        rollout=s.rollout,
        start_mode=s.start_mode,
        ctx_length=s.ctx_length,
        horizon=s.horizon,
        schedule=s.schedule,
        d=d,
        e=e,
        S=S,
        tau_seq=tau_seq,
        k_max=s.k_max,
        add_ctx_noise_std=getattr(s, "add_ctx_noise_std", 0.0),
    )

# ---------------------------
# Config
# ---------------------------

@dataclass(frozen=True)
class RealismConfig:
    # data
    B: int = 8
    T: int = 64
    H: int = 32
    W: int = 32
    C: int = 3
    pixels_per_step: int = 2
    size_min: int = 6
    size_max: int = 14
    hold_min: int = 4
    hold_max: int = 9
    diversify_data: bool = True

    # tokenizer / dynamics sizes
    patch: int = 4
    enc_n_latents: int = 16
    enc_d_bottleneck: int = 32
    d_model_enc: int = 64
    d_model_dyn: int = 128
    enc_depth: int = 8
    dec_depth: int = 8
    dyn_depth: int = 6
    n_heads: int = 4
    packing_factor: int = 2
    n_r: int = 4

    # schedule (realism): k_max=8; use σ-sampling + two bins {e=2,3}; bootstrap after warm-up
    k_max: int = 8
    sigma_sampling: bool = True
    multi_step_bins: bool = True
    use_bootstrap: bool = True
    bootstrap_start: int = 5_000  # warm-up steps with bootstrap disabled

    # train
    max_steps: int = 50_000
    log_every: int = 1_000
    lr: float = 3e-4

    # IO
    run_name: str = "realism_streaming_nodelta"
    tokenizer_ckpt: str = "/home/edward/projects/tiny_dreamer_4/logs/test/checkpoints"

# ---------------------------
# Main training loop
# ---------------------------

def run_realism(cfg: RealismConfig):
    out_dir = _ensure_dir(Path("./overfit_exps") / cfg.run_name)
    print(f"[setup] writing artifacts to: {out_dir.resolve()}")

    # Data iterator (streaming)
    next_batch = make_iterator(
        cfg.B, cfg.T, cfg.H, cfg.W, cfg.C,
        pixels_per_step=cfg.pixels_per_step,
        size_min=cfg.size_min, size_max=cfg.size_max,
        hold_min=cfg.hold_min, hold_max=cfg.hold_max,
        fg_min_color=0 if cfg.diversify_data else 128,
        fg_max_color=255 if cfg.diversify_data else 128,
        bg_min_color=0 if cfg.diversify_data else 255,
        bg_max_color=255 if cfg.diversify_data else 255,
    )

    # One batch to shape/init models & tokenizer restore
    init_rng = jax.random.PRNGKey(0)
    _, (frames_init, actions_init) = next_batch(init_rng)

    patch = cfg.patch
    num_patches = (cfg.H // patch) * (cfg.W // patch)
    D_patch = patch * patch * cfg.C
    k_max = cfg.k_max

    enc_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        n_heads=cfg.n_heads,
        depth=cfg.enc_depth,
        dropout=0.0,
        d_bottleneck=cfg.enc_d_bottleneck,
        mae_p_min=0.0, mae_p_max=0.0,
        time_every=4, latents_only_time=True,
    )
    dec_kwargs = dict(
        d_model=cfg.d_model_enc,
        n_heads=cfg.n_heads,
        depth=cfg.dec_depth,
        n_latents=cfg.enc_n_latents,
        n_patches=num_patches,
        d_patch=D_patch,
        dropout=0.0,
        mlp_ratio=4.0, time_every=4, latents_only_time=True,
    )
    n_s = cfg.enc_n_latents // cfg.packing_factor
    dyn_kwargs = dict(
        d_model=cfg.d_model_dyn,
        d_bottleneck=cfg.enc_d_bottleneck,
        d_spatial=cfg.enc_d_bottleneck * cfg.packing_factor,
        n_s=n_s, n_r=cfg.n_r,
        n_heads=cfg.n_heads, depth=cfg.dyn_depth,
        dropout=0.0, k_max=k_max,
        time_every=4, latents_only_time=False,
    )

    encoder = Encoder(**enc_kwargs)
    decoder = Decoder(**dec_kwargs)
    dynamics = Dynamics(**dyn_kwargs)

    patches_btnd = temporal_patchify(frames_init, patch)
    rng = jax.random.PRNGKey(0)
    enc_vars = encoder.init({"params": rng, "mae": rng, "dropout": rng}, patches_btnd, deterministic=True)
    fake_z = jnp.zeros((cfg.B, cfg.T, cfg.enc_n_latents, cfg.enc_d_bottleneck))
    dec_vars = decoder.init({"params": rng, "dropout": rng}, fake_z, deterministic=True)

    # Restore tokenizer params
    enc_vars, dec_vars, _ = load_pretrained_tokenizer(
        cfg.tokenizer_ckpt, rng=rng,
        encoder=encoder, decoder=decoder,
        enc_vars=enc_vars, dec_vars=dec_vars,
        sample_patches_btnd=patches_btnd,
    )

    mae_eval_key = jax.random.PRNGKey(777)

    # Build initial z1 to shape dynamics init
    z_btLd, _ = encoder.apply(enc_vars, patches_btnd, rngs={"mae": mae_eval_key}, deterministic=True)
    z1 = pack_bottleneck_to_spatial(z_btLd, n_s=n_s, k=cfg.packing_factor)
    emax = jnp.log2(k_max).astype(jnp.int32)
    step_idx = jnp.full((cfg.B, cfg.T), emax, dtype=jnp.int32)
    sigma_idx = jnp.full((cfg.B, cfg.T), k_max - 1, dtype=jnp.int32)
    dyn_vars = dynamics.init({"params": rng, "dropout": rng}, actions_init, step_idx, sigma_idx, z1)
    params = dyn_vars["params"]

    tx = optax.adam(cfg.lr)
    opt_state = tx.init(params)

    # Build train_step with static flags
    train_step = make_train_step(dict(
        sigma_sampling=cfg.sigma_sampling,
        multi_step_bins=cfg.multi_step_bins,
        use_bootstrap=cfg.use_bootstrap,
        patch=cfg.patch, n_s=n_s, k_max=k_max, packing_factor=cfg.packing_factor,
    ))

    # Streaming RNGs
    train_rng = jax.random.PRNGKey(2025)
    data_rng = jax.random.PRNGKey(12345)

    start_wall = time.time()
    for step in range(cfg.max_steps + 1):
        # Stream a fresh batch
        data_rng, batch_key = jax.random.split(data_rng)
        _, (frames, actions) = next_batch(batch_key)

        # Split master rng
        train_rng, master_key = jax.random.split(train_rng)

        # Warm-up → no bootstrap; after bootstrap_start → enable
        use_boot_now = cfg.use_bootstrap and (step >= cfg.bootstrap_start)

        params, opt_state, aux = train_step(
            encoder, decoder, dynamics, tx,
            params, opt_state, enc_vars, dec_vars, dyn_vars,
            frames, actions,
            patch=cfg.patch, n_s=n_s, k_max=k_max, packing_factor=cfg.packing_factor,
            sigma_sampling=cfg.sigma_sampling,
            multi_step_bins=cfg.multi_step_bins,
            use_bootstrap=use_boot_now,
            master_key=master_key,
        )

        if (step % cfg.log_every == 0) or (step == cfg.max_steps):
            pieces = [f"[train] step={step:06d}",
                      f"flow_mse={float(aux['flow_mse']):.6g}",
                      f"t={time.time()-start_wall:.1f}s"]
            if use_boot_now and "bootstrap_mse" in aux:
                pieces.append(f"boot_mse={float(aux['bootstrap_mse']):.6g}")
            print(" | ".join(pieces))

            # === Eval (AR) on a held-out batch ===
            val_rng = jax.random.PRNGKey(9999)
            _, (val_frames, val_actions) = next_batch(val_rng)

            dyn_vars_eval = with_params(dyn_vars, params)
            ctx_length = min(32, cfg.T - 1)
            regimes = _eval_regimes_for_realism(cfg, ctx_length=ctx_length)

            for tag, sconf in regimes:
                sconf.mae_eval_key = mae_eval_key
                sconf.rng_key = jax.random.PRNGKey(4242)

                t0 = time.time()
                pred_frames, floor_frames, gt_frames = sample_video(
                    encoder=encoder, decoder=decoder, dynamics=dynamics,
                    enc_vars=enc_vars, dec_vars=dec_vars, dyn_vars=dyn_vars_eval,
                    frames=val_frames, actions=val_actions, config=sconf,
                )
                dt = time.time() - t0

                HZ = sconf.horizon
                mse = jnp.mean((pred_frames[:, -HZ:] - gt_frames[:, -HZ:]) ** 2)
                psnr = psnr_from_mse(mse)
                print(f"[eval:{tag}] step={step:06d} | AR_hz={HZ} | MSE={float(mse):.6g} | PSNR={float(psnr):.2f} dB | {dt:.2f}s")

                # === Visualization: grid over entire batch ===
                # Convert to uint8 on host
                gt_np_all    = _to_uint8(gt_frames)      # (B, ctx+HZ, H, W, C)
                floor_np_all = _to_uint8(floor_frames)   # (B, ctx+HZ, H, W, C)
                pred_np_all  = _to_uint8(pred_frames)    # (B, ctx+HZ, H, W, C)

                # Build triptych per sample per timestep and tile into frames
                T_total = gt_np_all.shape[1]
                ncols = 1 if cfg.B <= 2 else min(2, cfg.B)
                grid_frames = []
                for t_idx in range(T_total):
                    trip_list = [
                        _stack_wide(gt_np_all[b, t_idx], floor_np_all[b, t_idx], pred_np_all[b, t_idx])
                        for b in range(cfg.B)
                    ]
                    grid_img = _tile_triptychs(trip_list, ncols=ncols, pad_color=0)
                    grid_frames.append(grid_img)

                # Paths
                tag_dir = _ensure_dir(out_dir / f"step_{step:06d}")
                gif_path = tag_dir / f"{tag}_grid.gif"
                mp4_path = tag_dir / f"{tag}_grid.mp4"

                # Write GIF
                imageio.mimsave(gif_path, grid_frames, duration=1/25, loop=1000)
                # Write MP4 (best-effort)
                try:
                    with imageio.get_writer(mp4_path, fps=25, codec="libx264", quality=8) as w:
                        for fr in grid_frames:
                            w.append_data(fr)
                except Exception as e:
                    print(f"[eval:{tag}] MP4 write skipped ({e}); GIF saved.")

                # Dump a sampler plan JSON next to the media
                plan = _plan_from_sconf(sconf)
                plan["step"] = int(step)
                plan["mse"] = float(mse)
                plan["psnr_db"] = float(psnr)
                plan_path = tag_dir / f"{tag}_plan.json"
                with open(plan_path, "w") as f:
                    json.dump(plan, f, indent=2)

                print(f"[eval:{tag}] wrote {gif_path.name}, "
                      f"{mp4_path.name if mp4_path.exists() else '(gif only)'}, "
                      f"and {plan_path.name} in {tag_dir}")

    # Save final config used
    (out_dir / "config.txt").write_text("\n".join([f"{k}={v}" for k, v in asdict(cfg).items()]))

# ---------------------------
# Entrypoint
# ---------------------------

if __name__ == "__main__":
    cfg = RealismConfig()
    print(f"Running realism config:\n  " + "\n  ".join([f"{k}={v}" for k,v in asdict(cfg).items()]))
    run_realism(cfg)
